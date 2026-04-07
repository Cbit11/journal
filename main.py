import os
import torch
import yaml
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier as dist_barrier, all_reduce, ReduceOp
from data.Custom_image_dataset import dataset
import datetime
import wandb
import argparse
from argparse import Namespace
from arch.model1 import main_block
from timm import utils
from torch.amp import autocast, GradScaler
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
torch.backends.cudnn.benchmark = True
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_norm_layer(layer_name):
    """Maps string names to PyTorch normalization classes."""
    if layer_name == 'LayerNorm' or layer_name == 'nn.LayerNorm':
        return nn.LayerNorm
    elif layer_name == 'BatchNorm':
        return nn.BatchNorm2d
    elif layer_name == 'Identity':
        return nn.Identity
    else:
        raise NotImplementedError(f"Normalization layer {layer_name} is not found")
def get_activation(act):
    if act=='GELU':
        return nn.GELU
    else:
        raise NotImplementedError(f"Activation function {act} is not found")
def load_config_and_parse_args():
    # --- 1. Initial Parser to get the config file path ---
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('-c', '--config', default='/home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA/options/journal_model_tiny.yaml', type=str, metavar='FILE',
                                help='YAML config file specifying default arguments')
    
    # Parse just the config file path, leaving other arguments for the main parser
    config_args, remaining_argv = config_parser.parse_known_args()

    defaults = {}
    if config_args.config and os.path.exists(config_args.config):
        print(f"Configuration loaded from YAML: {config_args.config}")
        with open(config_args.config, 'r') as f:
            # Load all YAML content into a single flat dictionary
            defaults = yaml.safe_load(f) or {}
    else:
        # If no config file is provided or found, this message is useful for debugging
        print("No valid YAML configuration file loaded. Using hardcoded defaults.")
    
    if isinstance(defaults.get('norm_layer'), str):
        defaults['norm_layer']= get_norm_layer(defaults['norm_layer'])
    main_parser= argparse.ArgumentParser()
    main_parser.add_argument('--train_hr_pth', type = str)
    main_parser.add_argument('--train_lr_pth', type = str)
    main_parser.add_argument('--val_hr_pth', type = str)
    main_parser.add_argument('--val_lr_pth', type = str)
    main_parser.set_defaults(**defaults)
    args = main_parser.parse_args(remaining_argv)
    return args
        
def load_checkpoint(chkpt_pth, model, optimizer, scheduler):
    # Map location is important for DDP to ensure it loads to the correct GPU
    current_device = torch.cuda.current_device()
    map_location = f'cuda:{current_device}'
    
    print(f"Loading checkpoint from {chkpt_pth} to {map_location}...")
    chkpt = torch.load(chkpt_pth, map_location=map_location)
    
    model.load_state_dict(chkpt['Model State'])

    if optimizer and 'optimizer_state' in chkpt:
        optimizer.load_state_dict(chkpt['optimizer_state'])
    if scheduler and 'scheduler_state' in chkpt:
        scheduler.load_state_dict(chkpt['scheduler_state'])
        
    iter = chkpt['Current Iteration']
    print(f"Resuming training from snapshot at iteration {iter}")
    return iter   # Return the NEXT epoch index
    
def save_checkpoint(current_iter, model, args, optimizer, scheduler):
    chkpt= {}
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        chkpt['Model State']= model.module.state_dict()
        chkpt['Current Iteration']= current_iter
        chkpt['optimizer_state']= optimizer.state_dict()
        chkpt['scheduler_state'] = scheduler.state_dict()
        torch.save(chkpt, os.path.join(args.checkpoint_folder,f'chkpt_{current_iter}_{args.name}.pt'))
        torch.save(chkpt, os.path.join(args.checkpoint_folder, f'latest_checkpoint_{args.name}.pt'))
        print(f"Iteration: {current_iter} | Training snapshot saved.")

def tiled_inference(model,img_lr, scale=2, tile_size=256, overlap=32):
    b, c, h, w = img_lr.shape
    out_h, out_w = h * scale, w * scale
    output = torch.zeros((b, c, out_h, out_w), device=img_lr.device)
    output_count = torch.zeros((b, c, out_h, out_w), device=img_lr.device)

    # 2. Define the stride (step size)
    stride = tile_size - overlap
    
    # 3. Create a grid of starting coordinates
    h_starts = list(range(0, h, stride))
    w_starts = list(range(0, w, stride))
    
    # Ensure the last tile covers the edge (it might overlap more than others)
    h_starts = [min(x, h - tile_size) for x in h_starts]
    w_starts = [min(x, w - tile_size) for x in w_starts]
    
    # Remove duplicates if image is smaller than tile_size or stride aligns perfectly
    h_starts = sorted(list(set(h_starts)))
    w_starts = sorted(list(set(w_starts)))

    model.eval()
    
    # 4. Loop through all tiles
    with torch.no_grad():
        for y in h_starts:
            for x in w_starts:
                # Crop the Input (LR)
                lr_patch = img_lr[:, :, y : y + tile_size, x : x + tile_size]
                
                # Inference
                # Use Automatic Mixed Precision (AMP) for speed if available
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    sr_patch = model(lr_patch)
                
                # Calculate Output Coordinates (HR)
                y_out = y * scale
                x_out = x * scale
                out_tile_size = tile_size * scale
                
                # Accumulate the result in the output tensor
                output[:, :, y_out : y_out + out_tile_size, x_out : x_out + out_tile_size] += sr_patch
                
                # Accumulate the count (for averaging later)
                output_count[:, :, y_out : y_out + out_tile_size, x_out : x_out + out_tile_size] += 1

    # 5. Average the overlapping regions
    output = output / output_count
    
    return output

def main():
    utils.setup_default_logging()
    local_rank = int(os.environ.get("SLURM_LOCALID", "0")) 
    rank = int(os.environ.get("SLURM_PROCID","0"))
    world_size = int(os.environ.get("SLURM_NTASKS","1"))
    os.environ["WANDB_START_METHOD"] = "thread"
    current_device = local_rank
    torch.cuda.set_device(local_rank)
    init_process_group(backend='nccl',world_size=world_size, rank=rank, timeout= datetime.timedelta(seconds=7200))
    args = load_config_and_parse_args()
    device = torch.device(f"cuda:{current_device}") 
    model = main_block(
                 img_size= args.img_size, 
                 patch_size= args.patch_size, 
                 in_chans= args.in_chans, 
                 dims= args.dims,
                 kernel_size= args.kernel_size, 
                 dilation= args.dilation, 
                 num_heads= args.num_heads,
                 num_blocks= args.num_blocks,
                 stride=args.stride,
                 downsample=args.downsample, 
                 upscale=args.upscale,
                 drop_rate=args.drop_rate,
                 drop_path_rate=args.drop_path_rate,
                 window_size= args.window_size,
                 overlap_ratio= args.overlap_ratio, 
                 ape = args.ape,
                 upsampler= args.upsampler,
                 d_state=args.d_state,
                 d_conv=args.d_conv,
                 expand=args.expand, 
                 norm_layer= args.norm_layer, 
                 resi_connection=args.resi_connection, 
                 img_range=args.img_range
    ).to(device)
    moedl = torch.compile(model)
    model = DDP(model, device_ids= [local_rank])
    #Build Optimizers and loss functions and schedulers
    loss_fn = nn.L1Loss()
    optimizer= torch.optim.Adam(params= model.parameters(), lr = args.lr, betas= (args.betas))
    main_scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones)
    if args.warmup_iter is not None:
        warmup_scheduler= torch.optim.lr_scheduler.LinearLR(optimizer, total_iters= args.warmup_iter)
        lr_scheduler= torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_iter])
    else: 
        lr_scheduler= main_scheduler
    latest_ckpt_path = os.path.join(args.checkpoint_folder, f'latest_checkpoint_{args.name}.pt')
    current_iter= 0
    if os.path.exists(latest_ckpt_path):
        # We only print on rank 0 to avoid clutter
        if local_rank == 0:
            print(f"Found checkpoint at {latest_ckpt_path}. Resuming...")
        # Load logic
        current_iter = load_checkpoint(latest_ckpt_path, model.module, optimizer, lr_scheduler)
        print(f"Resuming traning from iter: {current_iter}")
        dist_barrier() # Ensure all ranks have loaded before continuing
    else:
        if args.resume_training:
            current_iter = load_checkpoint(args.checkpoint_path, model.module, optimizer, lr_scheduler)
            dist_barrier()
            if local_rank ==0:
                print(f"Loading with the last saved checkpoint")
    # Build Dataset and dataloaders
    train_data = dataset(args.train_hr_pth, args.train_lr_pth, train_type='train', scale = args.scale, gt_size=args.gt_size)
    train_sampler = DistributedSampler(train_data, shuffle=args.train_shuffle)
    train_loader= DataLoader(
        train_data, 
        batch_size= args.train_batch_size, 
        sampler= train_sampler,
        num_workers= 4,
        pin_memory= True
    )
    val_data= dataset(args.val_hr_pth, args.val_lr_pth,train_type='val', scale= args.scale, gt_size=args.gt_size)
    val_sampler= DistributedSampler(val_data, shuffle=args.val_shuffle)
    val_loader= DataLoader(
        val_data, 
        batch_size= args.val_batch_size, 
        sampler= val_sampler,
        num_workers= 0,
        pin_memory= True
    )
    if local_rank == 0:
        run_id_file = os.path.join(args.checkpoint_folder, "wandb_run_id.txt")
        if os.path.exists(run_id_file):
            with open(run_id_file, "r") as f:
                run_id = f.read().strip()
            run = wandb.init(project=args.name, id=run_id, resume="allow", config=args)
        else:
            run = wandb.init(project=args.name, config=args)
            with open(run_id_file, "w") as f:
                f.write(run.id)
    total_iteration = args.Iteration
    steps_per_epoch = len(train_loader)
    epoch = current_iter // steps_per_epoch
    print_freq= 500
    accumulation_steps= 2
    val_freq= 1000
    running_loss= torch.zeros(1, device= device)
    step_count = 0
    optimizer.zero_grad()
    micro_step_ct= 0
    while current_iter< total_iteration:
        epoch+=1
        train_sampler.set_epoch(epoch) 
        model.train()
        # Train Loop 

        for batch, data in enumerate(train_loader):
            micro_step_ct+=1
            do_sync = (micro_step_ct) % accumulation_steps == 0
            gt = data['gt'].to(device, non_blocking=True)
            lr = data['lq'].to(device, non_blocking=True)
            my_context = model.no_sync() if (not do_sync) else torch.enable_grad()
            with my_context: 
                with autocast('cuda', dtype=torch.bfloat16):
                    pred = model(lr)
                    loss = loss_fn(pred, gt)
                    loss = loss / accumulation_steps
                loss.backward()
            running_loss += loss.detach() * accumulation_steps
            step_count += 1
            if do_sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                current_iter+=1

                if current_iter% print_freq==0:
                    avg_loss = running_loss / step_count
                    all_reduce(avg_loss, op=ReduceOp.AVG)
                    if local_rank ==0: 
                        print(f"[Current Iteration {current_iter}] Train Loss: {avg_loss.item():.4f}")
                        run.log({"Train Loss":avg_loss.item()})
                        latest_ckpt = {
                            'Model State': model.module.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': lr_scheduler.state_dict(),
                            'Current Iteration': current_iter
                            }
                        torch.save(latest_ckpt, os.path.join(args.checkpoint_folder, f'latest_checkpoint_{args.name}.pt'))
                        print(f"Current Iteration {current_iter} finished. 'latest_checkpoint_{args.name}.pt' updated.")
                    running_loss = torch.zeros(1, device=device)
                    step_count = 0
                if current_iter%val_freq==0:
                    model.eval()
                    val_loss = torch.zeros(1, device=device)
                    total_psnr = torch.zeros(1, device=device)
                    total_ssim = torch.zeros(1, device=device)
                    num_val_batches = torch.tensor(0, device=device) # Count batches
                    with torch.no_grad():
                        for vdata in val_loader:
                            gt = vdata['gt'].to(device,non_blocking=True)
                            lr = vdata['lq'].to(device,non_blocking=True)
                            pred = tiled_inference(model.module, lr, args.scale, tile_size=128)
                            pred= pred.float()
                            pred.clamp_(0, 1)
                            val_loss+=loss_fn(pred, gt)
                 # Make sure these functions return tensors
                            total_psnr += calculate_psnr_pt(pred, gt, crop_border=0, test_y_channel=True)
                            total_ssim += calculate_ssim_pt(pred, gt, crop_border=0, test_y_channel=True)
                            num_val_batches += 1

                    all_reduce(val_loss, op=ReduceOp.SUM)
                    all_reduce(total_psnr, op=ReduceOp.SUM)
                    all_reduce(total_ssim, op=ReduceOp.SUM)
                    all_reduce(num_val_batches, op=ReduceOp.SUM) # Total batches across all GPUs

                    avg_val_loss = val_loss / num_val_batches
                    avg_psnr = total_psnr / num_val_batches
                    avg_ssim = total_ssim / num_val_batches

                    if local_rank == 0:
                        print(f"Validation | Loss: {avg_val_loss.item():.4f} | PSNR: {avg_psnr.item():.2f} | SSIM: {avg_ssim.item():.4f}")
                        run.log({
                        "Validation loss": avg_val_loss.item(), 
                        "PSNR": avg_psnr.item(), 
                        "SSIM": avg_ssim.item()
                    })
                        save_checkpoint(current_iter, model, args, optimizer, lr_scheduler)

                model.train() # Switch back to train mode
                if current_iter >= total_iteration:
                    break

            if current_iter >= total_iteration:
                break

            #Validation Block 
        
    print(f"Completed training for {total_iteration}: Success!!!!")
    if local_rank == 0:
        if run:
            wandb.finish()
    destroy_process_group()
if __name__== "__main__":
    main()