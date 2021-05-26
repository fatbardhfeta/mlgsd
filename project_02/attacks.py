import torch


def fast_gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, epsilon: float, norm: str = "2",
                         loss_fn=torch.nn.functional.cross_entropy):
    """
    Perform a single-step projected gradient attack on the input x.
    Parameters
    ----------
    logits: torch.Tensor of shape [B, K], where B is the batch size and K is the number of classes.
        The logits for each sample in the batch.
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the number of channels, and N is the image
       dimension.
       The input batch of images. Note that x.requires_grad must have been active before computing the logits
       (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    epsilon: float
        The desired strength of the perturbation. That is, the perturbation (before clipping) will have a norm of
        exactly epsilon as measured by the desired norm (see argument: norm).
    norm: str, can be ["1", "2", "inf"]
        The norm with which to measure the perturbation. E.g., when norm="1", the perturbation (before clipping)
         will have a L_1 norm of exactly epsilon (see argument: epsilon).
    loss_fn: function
        The loss function used to construct the attack. By default, this is simply the cross entropy loss.

    Returns
    -------
    torch.Tensor of shape [B, C, N, N]: the perturbed input samples.

    """
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

    ##########################################################
    x.requires_grad = True

    

    #predictions = torch.argmax(logits, 1)
    loss = loss_fn(logits, y)

    loss.retain_grad()

    data_grad = loss.backward(retain_graph=True) #gradients of loss

    data_grad = loss.grad  
    
    #check if any projection is needed and apply it to step variable to rescale it into our domain
    scaled_step = 0
    lr = 0.05
    if(norm == "1"):

        distance = torch.norm(data_grad, p = 1)
        scaled_step = lr * data_grad

        if distance >= epsilon:
            scaled_step = lr*( data_grad / distance)


    elif( norm == "2"):

        distance = torch.norm(data_grad, p = 2)
        scaled_step = lr * data_grad

        if distance >= epsilon:
            scaled_step = lr*( data_grad / distance)
    
    elif( norm == "inf"):

        distance = torch.norm(data_grad, float('inf'))
        scaled_step = lr * data_grad

        if distance >= epsilon:
            scaled_step = epsilon* torch.sign(data_grad)

    else:
        print("The norm is not correct")

    print(data_grad.size())
    print(data_grad)
    
    x_pert = torch.add(x, scaled_step)

    x_pert = torch.clip(x_pert,0,1)
    ##########################################################

    return x_pert.detach()
