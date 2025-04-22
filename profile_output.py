print_rank_0(f"PROFILE CONFIG: rank={torch.distributed.get_rank()} start={args.profile_step_start}, end={args.profile_step_end}, ranks={args.profile_ranks}")


print_rank_0(f"[DEBUG] PROFILE CONFIG: rank={torch.distributed.get_rank()}, "
                     f"start={args.profile_step_start}, "
                     f"end={args.profile_step_end}, "
                     f"active={args.profile_step_end - args.profile_step_start}, "
                     f"ranks={args.profile_ranks}, "
                     f"tensorboard_dir={args.tensorboard_dir}")
