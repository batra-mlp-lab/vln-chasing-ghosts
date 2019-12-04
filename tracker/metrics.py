import torch


def nav_error(target, prediction, args):
    """ Calculate Navigation Error (in metre) between 2 points that are in metres

    Args:
        target (torch[.cuda].LongTensor): (batch_size, timesteps, 2)
        prediction (torch[.cuda].LongTensor): (batch_size, timesteps, 2)
        args

    Returns:
        error (torch[.cuda].FloatTensor): Timestep-wise nav error
                                          shape: (batch_size, timesteps)
        averaged_error (float): Nav Error averaged over timesteps
                                shape: (batch_size)
    """
    error_yx_metre = prediction - target
    error = (error_yx_metre[:, :, 0]**2 + error_yx_metre[:, :, 1]**2)**0.5

    averaged_error = torch.sum(error, dim=1).float()/ args.timesteps

    return error, averaged_error


def success_rate(target, prediction, args, success_threshold=3):
    """ Calculate Success Rate between 2 points that are in metres

    Args:
        target (torch[.cuda].LongTensor): (batch_size, 2)
        prediction (torch[.cuda].LongTensor): (batch_size, 2)
        args
        success_threshold (int, optional)

    Returns:
        timestep_success_rate (torch[.cuda].FloatTensor): Timestep-wise success rate
                                                          in [0,1]. shape: (batch_size, timesteps)
        average_success_rate (float): Success rate averaged over timesteps
                                      shape: (batch_size)
    """
    error_yx_metre = prediction - target
    error = (error_yx_metre[:, :, 0]**2 + error_yx_metre[:, :, 1]**2)**0.5

    success = torch.where(error <= success_threshold,
                torch.tensor(1, device=args.device), torch.tensor(0, device=args.device))

    average_success_rate = torch.sum(success, dim=1).float() / args.timesteps

    return success, average_success_rate


def map_coverage(map_mask):
    """
    Compute how much of the total map (map_range_y, map_range_x) is explored
    Args:
        map_mask (torch[.cuda].FloatTensor): [description]
            shape: (batch_size, timesteps, size_y, size_x)
    Returns:
        timestep_coverage (torch[.cuda].FloatTensor): shape: (batch_size, timesteps)
        average_coverage (torch[.cuda].FloatTensor): shape: (batch_size)
    """
    timestep_coverage = torch.mean(map_mask, dim=[2, 3])
    average_coverage = torch.mean(map_mask, dim=[1, 2, 3])
    return timestep_coverage, average_coverage


def goal_seen_rate(map_mask, goal_pos, args):
    """
    Compute how many times is the goal seen in the map_mask

    Args:
        map_mask (torch[.cuda].FloatTensor): shape: (batch_size, timesteps, map_range_y, map_range_x)
        goal_pos (torch[.cuda].FloatTensor): shape: (batch_size, 2)
        args

    Returns:
        goal_seen (torch[.cuda].FloatTensor): shape: (batch_size, timesteps)
        average_goal_seen (torch[.cuda].FloatTensor): shape: (batch_size)
    """
    batch_size = map_mask.shape[0]
    timesteps = map_mask.shape[1]
    goal_seen = torch.zeros(batch_size, timesteps, dtype=torch.float, device=args.device)
    for n in range(batch_size):
        for t in range(timesteps):
            x = goal_pos[n][0]
            y = goal_pos[n][1]
            if map_mask[n][t][y][x] == 1:
                goal_seen[n][t] = 1
    average_goal_seen = torch.mean(goal_seen, dim=1)
    return goal_seen, average_goal_seen
