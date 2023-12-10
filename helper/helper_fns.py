def count_parameters(model):
    active_count = 0
    frozen_count = 0
    for p in model.parameters():
            if p.requires_grad:
                active_count += p.numel()
            else:
                frozen_count += p.numel()
            
    return (active_count, frozen_count)