import torch
import torch.nn.functional as F

def dice_single_cf_batch(
    model, x, y_target,
    lambda_proximity=0.1, gamma_diversity=0.0,
    prev_zs=None, num_steps=500, lr=0.01,
    loss_type='hinge',  # 'hinge' or 'bce'
    prox_type='l1',     # 'l1' or 'mad'
    mad=None            # 1D tensor of shape (D,) if prox_type=='mad'
):
    """
    Batched single counterfactual generation with configurable losses.

    Args:
        model: PyTorch model returning a single logit per sample.
        x: tensor (B, D) of original inputs.
        y_target: tensor (B,) target labels {0,1}.
        lambda_proximity: weight for proximity loss.
        gamma_diversity: weight for diversity.
        prev_zs: list of previous z tensors (B, D).
        num_steps: number of optimization steps.
        lr: learning rate.
        loss_type: 'hinge' or 'bce'.
        prox_type: 'l1' or 'mad'.
        mad: median absolute deviation per feature for 'mad' prox.
    Returns:
        z: tensor (B, D) of counterfactuals.
    """
    B, D = x.shape
    # init z from x
    z = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    # define classification loss
    if loss_type == 'bce':
        clf_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    elif loss_type == 'hinge':
        clf_criterion = None
    else:
        raise ValueError("loss_type must be 'bce' or 'hinge'")

    for step in range(num_steps):
        optimizer.zero_grad()
        logits = model(z).view(-1)

        # classification loss
        if loss_type == 'bce':
            clf_loss = clf_criterion(logits, y_target.float())
        else:  # hinge
            t = 2 * y_target.float() - 1
            clf_loss = torch.clamp(1 - t * logits, min=0).mean()

        # proximity loss
        if prox_type == 'l1':
            prox_loss = torch.norm(z - x, p=1, dim=1).mean()
        elif prox_type == 'mad':
            if mad is None:
                raise ValueError("`mad` tensor must be provided for prox_type='mad'.")
            # avoid zero division
            mad_safe = mad.clone()
            mad_safe[mad_safe == 0] = 1.0
            #if step==1:
            #    print("mad_safe.shape", mad_safe.shape, mad_safe)
            #    print("torch.abs(z - x)z", torch.abs(z - x))
            #    print("mad_safe.unsqueeze(0)", mad_safe.unsqueeze(0))
            # scaled abs deviation per feature: (B, D)
            scaled = torch.abs(z - x) / mad_safe.unsqueeze(0)
            # sum over features and average by number of continuous features D
            per_sample = scaled.sum(dim=1) / D
            #if step==1:
            #    print("per_sample.shape", per_sample.shape, per_sample)
            # then mean over batch
            prox_loss = per_sample.mean()
        else:
            raise ValueError("prox_type must be 'l1' or 'mad'.")

        loss = clf_loss + lambda_proximity * prox_loss

        # diversity term
        if prev_zs and gamma_diversity > 0:
            div_term = 0.0
            for prev in prev_zs:
                div_term += torch.norm(z - prev, p=1, dim=1).mean()
            loss = loss - gamma_diversity * div_term

        loss.backward()
        optimizer.step()

    return z


def dice_cf_set_batch(
    model, x, logits,
    K=1, lambda_proximity=0.1, gamma_diversity=0.1,
    num_steps=500, lr=0.01,
    loss_type='hinge', prox_type='l1', mad=None
):
    """
    Greedy generation of K counterfactual batches with configurable losses.

    Args:
        model: PyTorch model returning a logit per sample.
        x: (B, D) input batch.
        logits: (B,) original predictions.
        K: number of counterfactual sets.
        loss_type, prox_type, mad: passed through.
    Returns:
        List of K tensors, each shape (B, D).
    """
    if prox_type == "mad":
        median = x.median(dim=0).values
        mad = (x.sub(median).abs().median(dim=0).values)
    if logits == None: 
        logits = model(x)
    y = (logits > 0).long()
    y_target = 1 - y
    cfs = []
    for _ in range(K):
        z = dice_single_cf_batch(
            model, x, y_target,
            lambda_proximity=lambda_proximity,
            gamma_diversity=gamma_diversity,
            prev_zs=cfs,
            num_steps=num_steps,
            lr=lr,
            loss_type=loss_type,
            prox_type=prox_type,
            mad=mad
        )
        cfs.append(z)
    return cfs



# Example usage:
if __name__ == "__main__":
    B, D = 128, 9
    class Dummy(torch.nn.Module):
        def __init__(self,D): super().__init__(); self.lin = torch.nn.Linear(D,1)
        def forward(self,x): return self.lin(x).squeeze(1)
    model = Dummy(D)
    x = torch.randn(B, D)
    logits = model(x)
    y = (logits > 0).long()
    print(y)
    # compute per-feature MAD: median(|x - median(x)|)
    median = x.median(dim=0).values
    mad = (x.sub(median).abs().median(dim=0).values)
    print("median.shape", median.shape, median)
    print("mad.shape", mad.shape, mad)
    # get 3 CF sets using hinge + MAD prox
    cfs = dice_cf_set_batch(
        model, x, y,
        K=1,
        loss_type='hinge', prox_type='mad', mad=mad
    )
    for i, z in enumerate(cfs, 1):
        print(f"CF batch {i}:", z.shape)
        if len(cfs) == 1:
            print(z)
            print(model(z))
