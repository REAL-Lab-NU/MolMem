from verl.trainer.ppo.core_algos import *
import math

# supported by Kangrui Wang
def compute_bi_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        reward_mask = token_level_rewards.bool()
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()
        
        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions=reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]
                
                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]
            
            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns


def compute_rank_based_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    best_molecule_scores: torch.Tensor,
    index: np.ndarray,
    cumulative_weight: float = 0.5,
    rank_temperature: float = 1.0
):
    """
    Compute rank-based GRPO advantages using both cumulative rewards and best molecule scores.
    
    Args:
        token_level_rewards: (batch_size, seq_len) Token-level rewards
        response_mask: (batch_size, seq_len) Mask for response tokens
        best_molecule_scores: (batch_size,) Best molecule scores for each trajectory
        index: (batch_size,) Trajectory indices for grouping
        cumulative_weight: Weight for cumulative reward ranking (0.0-1.0)
        rank_temperature: Temperature for rank-based advantage scaling
        
    Returns:
        advantages: (batch_size, seq_len) Computed advantages
        returns: (batch_size, seq_len) Computed returns
    """
    with torch.no_grad():
        batch_size = token_level_rewards.shape[0]
        
        # Compute cumulative rewards per trajectory
        cumulative_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # (batch_size,)
        
        # Convert best_molecule_scores to tensor if needed
        if isinstance(best_molecule_scores, np.ndarray):
            best_molecule_scores = torch.from_numpy(best_molecule_scores).float()
        if isinstance(best_molecule_scores, list):
            best_molecule_scores = torch.tensor(best_molecule_scores).float()
        
        # Ensure same device
        best_molecule_scores = best_molecule_scores.to(cumulative_rewards.device)
        
        # Group trajectories by index (for comparison within same initial molecule)
        unique_indices = np.unique(index)
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        for idx in unique_indices:
            # Find trajectories with this index
            mask = index == idx
            if not np.any(mask):
                continue
                
            indices = np.where(mask)[0]
            
            if len(indices) < 2:
                # If only one trajectory for this molecule, set advantage to 0
                continue
            
            # Extract scores for this group
            group_cumulative = cumulative_rewards[indices]  # (group_size,)
            group_best_molecule = best_molecule_scores[indices]  # (group_size,)
            
            # Compute ranks (higher score = better rank)
            # argsort gives indices that would sort the array, we want reverse for ranking
            cumulative_ranks = torch.argsort(torch.argsort(group_cumulative, descending=True)) + 1  # Rank 1 = best
            best_molecule_ranks = torch.argsort(torch.argsort(group_best_molecule, descending=True)) + 1
            
            # Normalize ranks to [0, 1] range, where 0 = best rank
            cumulative_ranks_norm = (cumulative_ranks - 1) / (len(indices) - 1) if len(indices) > 1 else torch.zeros_like(cumulative_ranks)
            best_molecule_ranks_norm = (best_molecule_ranks - 1) / (len(indices) - 1) if len(indices) > 1 else torch.zeros_like(best_molecule_ranks)
            
            # Combine ranks with weighting
            combined_ranks = cumulative_weight * cumulative_ranks_norm + (1 - cumulative_weight) * best_molecule_ranks_norm
            
            # Convert to advantages: lower rank (better performance) = higher advantage
            # Apply temperature scaling
            rank_advantages = -(combined_ranks - 0.5) / rank_temperature  # Center around 0, scale by temperature
            
            # Assign advantages to corresponding positions
            for i, traj_idx in enumerate(indices):
                # Spread the advantage across all response tokens for this trajectory
                response_positions = response_mask[traj_idx].bool()
                advantages[traj_idx, response_positions] = rank_advantages[i]
                
                # Returns are just the advantages (no value function in GRPO)
                returns[traj_idx, response_positions] = rank_advantages[i]
        
        # Normalize advantages globally (across all trajectories)
        advantages_flat = advantages[response_mask.bool()]
        if len(advantages_flat) > 1 and advantages_flat.std() > 1e-8:
            advantages_mean = advantages_flat.mean()
            advantages_std = advantages_flat.std()
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
            returns = (returns - advantages_mean) / (advantages_std + 1e-8)
    
    return advantages, returns


def compute_lambda_weights(labels: torch.Tensor, predicted_ranks: torch.Tensor):
    """
    Compute Lambda weights for LiPO-λ loss based on DCG-style ranking.
    
    Args:
        labels: (K,) Preference labels/scores for each response
        predicted_ranks: (K,) Predicted ranks (1-indexed, 1=best)
        
    Returns:
        lambda_weights: (K, K) Matrix of weights for all pairs
    """
    K = len(labels)
    lambda_weights = torch.zeros(K, K, device=labels.device)
    
    # Gain function: G_i = 2^ψ_i - 1 (using labels as ψ)
    gains = 2 ** labels - 1
    
    # Rank discount function: D(τ(i)) = log(1 + τ(i))
    discounts = torch.log(1 + predicted_ranks.float())
    
    for i in range(K):
        for j in range(K):
            if i != j:
                # Lambda weight: |G_i - G_j| * |1/D(τ(i)) - 1/D(τ(j))|
                gain_diff = torch.abs(gains[i] - gains[j])
                discount_diff = torch.abs(1.0 / discounts[i] - 1.0 / discounts[j])
                lambda_weights[i, j] = gain_diff * discount_diff
    
    return lambda_weights


def compute_pair_lambda_weight_simple(
    better_env_reward: float,
    worse_env_reward: float,
    better_rank: int,
    worse_rank: int
) -> float:
    """
    Simplified Lambda weight computation following the paper formula exactly.
    
    Args:
        better_env_reward: Environment reward r_t of the better turn
        worse_env_reward: Environment reward r_t of the worse turn  
        better_rank: Rank position ρ_t of better turn within trajectory (1-indexed)
        worse_rank: Rank position ρ_t of worse turn within trajectory (1-indexed)
        
    Returns:
        lambda_weight: Float weight for this pair
    """
    # Gain function: G(r) = 2^r - 1
    gain_better = 2 ** better_env_reward - 1
    gain_worse = 2 ** worse_env_reward - 1
    
    # Rank discount function: D(ρ) = log(1 + ρ)
    discount_better = math.log(1 + better_rank)
    discount_worse = math.log(1 + worse_rank)
    
    # Lambda weight: |G(r_i) - G(r_j)| * |1/D(ρ_i) - 1/D(ρ_j)|
    gain_diff = abs(gain_better - gain_worse)
    discount_diff = abs(1.0 / (discount_better + 1e-8) - 1.0 / (discount_worse + 1e-8))
    lambda_weight = gain_diff * discount_diff
    
    # Clamp to reasonable range
    return max(0.01, min(lambda_weight, 5.0))


def build_preference_pairs(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    best_molecule_scores: torch.Tensor = None,
    group_ids: np.ndarray = None,
    mode: str = "both",  # "intra", "inter", or "both"
    inter_weight_cumulative: float = 0.5,  # Weight for cumulative vs best score in inter-trajectory
    max_intra_pairs_per_trajectory: int = 6,  # 限制每个轨迹的intra pairs数量
    max_inter_pairs_per_molecule: int = 20   # 限制每个分子的inter pairs数量
):
    """
    Build preference pairs for LiPO-λ loss computation.
    
    Args:
        token_level_rewards: (batch_size, seq_len) Token-level rewards
        response_mask: (batch_size, seq_len) Mask for response tokens
        best_molecule_scores: (batch_size,) Best molecule scores for inter-trajectory comparison
        group_ids: (batch_size,) Group IDs for grouping trajectories by initial molecule
        mode: Type of preference pairs to build
        
    Returns:
        preference_data: Dict containing pair information and labels
    """
    batch_size = token_level_rewards.shape[0]
    device = token_level_rewards.device
    
    # Compute cumulative rewards per trajectory
    cumulative_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # (batch_size,)
    
    preference_pairs = []
    preference_labels = []
    preference_scores = []
    
    if mode in ["intra", "both"]:
        # Intra-trajectory preferences: compare different turns within same trajectory
        # Find positions with any non-zero rewards (including negative)
        reward_mask = token_level_rewards != 0
        
        for batch_idx in range(batch_size):
            turn_positions = reward_mask[batch_idx].nonzero(as_tuple=True)[0]
            turn_rewards = token_level_rewards[batch_idx, turn_positions]
            
            if len(turn_positions) >= 2:
                # 收集所有可能的pairs
                trajectory_pairs = []
                for i in range(len(turn_rewards)):
                    for j in range(i + 1, len(turn_rewards)):
                        if turn_rewards[i] != turn_rewards[j]:  # Skip equal rewards
                            reward_diff = abs(turn_rewards[i].item() - turn_rewards[j].item())
                            trajectory_pairs.append((i, j, reward_diff))
                
                # 如果pairs太多，选择差异最大的pairs
                if len(trajectory_pairs) > max_intra_pairs_per_trajectory:
                    # 按奖励差异排序，选择差异最大的pairs
                    trajectory_pairs.sort(key=lambda x: x[2], reverse=True)
                    trajectory_pairs = trajectory_pairs[:max_intra_pairs_per_trajectory]
                
                # 创建preference pairs
                for i, j, _ in trajectory_pairs:
                    if turn_rewards[i] > turn_rewards[j]:
                        better_idx, worse_idx = i, j
                        preference_direction = 1
                    else:
                        better_idx, worse_idx = j, i
                        preference_direction = -1
                    
                    preference_pairs.append((batch_idx, batch_idx, better_idx, worse_idx))  # 统一使用0-based索引
                    preference_labels.append(preference_direction)
                    preference_scores.append((turn_rewards[better_idx].item(), turn_rewards[worse_idx].item()))
    
    if mode in ["inter", "both"] and best_molecule_scores is not None and group_ids is not None:
        # Inter-trajectory preferences: compare different trajectories for same initial molecule
        # Combine cumulative rewards and best molecule scores with weighting
        unique_groups = np.unique(group_ids)
        
        for group_id in unique_groups:
            group_mask = group_ids == group_id
            group_indices = np.where(group_mask)[0]
            
            if len(group_indices) >= 2:
                group_best_scores = best_molecule_scores[group_indices]
                group_cumulative_scores = cumulative_rewards[group_indices]
                
                # Normalize scores to [0, 1] for fair combination
                if len(group_indices) > 1:
                    best_min, best_max = group_best_scores.min(), group_best_scores.max()
                    cum_min, cum_max = group_cumulative_scores.min(), group_cumulative_scores.max()
                    
                    if best_max > best_min:
                        norm_best = (group_best_scores - best_min) / (best_max - best_min)
                    else:
                        norm_best = torch.ones_like(group_best_scores)
                    
                    if cum_max > cum_min:
                        norm_cum = (group_cumulative_scores - cum_min) / (cum_max - cum_min)
                    else:
                        norm_cum = torch.ones_like(group_cumulative_scores)
                    
                    # Combined score: weighted average of normalized scores
                    combined_scores = inter_weight_cumulative * norm_cum + (1 - inter_weight_cumulative) * norm_best
                else:
                    combined_scores = group_best_scores  # Fallback for single trajectory
                
                # Create pairs within this group based on combined scores
                group_pairs = []
                for i in range(len(group_indices)):
                    for j in range(i + 1, len(group_indices)):
                        idx_i, idx_j = group_indices[i], group_indices[j]
                        combined_score_i, combined_score_j = combined_scores[i], combined_scores[j]
                        
                        if combined_score_i != combined_score_j:  # Skip equal scores
                            score_diff = abs(combined_score_i - combined_score_j)
                            group_pairs.append((i, j, score_diff, idx_i, idx_j, combined_score_i, combined_score_j))
                
                # 如果pairs太多，选择分数差异最大的pairs
                if len(group_pairs) > max_inter_pairs_per_molecule:
                    # 按分数差异排序，选择差异最大的pairs
                    group_pairs.sort(key=lambda x: x[2], reverse=True)
                    group_pairs = group_pairs[:max_inter_pairs_per_molecule]
                
                # 创建preference pairs
                for i, j, score_diff, idx_i, idx_j, combined_score_i, combined_score_j in group_pairs:
                    if combined_score_i > combined_score_j:
                        better_idx, worse_idx = idx_i, idx_j
                        preference_direction = 1
                    else:
                        better_idx, worse_idx = idx_j, idx_i
                        preference_direction = -1
                    
                    preference_pairs.append((better_idx, worse_idx, 0, 0))  # 0,0 for trajectory-level
                    preference_labels.append(preference_direction)
                    # Store combined scores for consistency with loss computation
                    better_combined = max(combined_score_i, combined_score_j)
                    worse_combined = min(combined_score_i, combined_score_j)
                    preference_scores.append((better_combined.item(), worse_combined.item()))
    
    return {
        "pairs": preference_pairs,
        "labels": preference_labels,
        "scores": preference_scores,
        "device": device,
        "batch_size": batch_size,  # 添加batch_size用于计算平均指标
        "token_level_rewards": token_level_rewards,  # 添加这个用于intra-trajectory比较
        "inter_weight_cumulative": inter_weight_cumulative  # 添加这个用于inter-trajectory loss计算
    }


def compute_lipo_lambda_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    preference_data: dict,
    beta: float = 0.05,
    temperature: float = 1.0,
    intra_weight: float = 1.0,
    inter_weight: float = 1.0,
    use_lambda_weights: bool = True,
    intra_comparison_mode: str = "cumulative",  # "cumulative" or "stepwise"
    best_molecule_scores: torch.Tensor = None  # 添加这个参数用于inter-trajectory计算
):
    """
    Compute LiPO-λ preference loss.
    
    Args:
        log_probs: (batch_size, seq_len) Current policy log probabilities
        old_log_probs: (batch_size, seq_len) Old policy log probabilities  
        ref_log_probs: (batch_size, seq_len) Reference policy log probabilities
        response_mask: (batch_size, seq_len) Mask for response tokens
        preference_data: Dict from build_preference_pairs()
        beta: Scaling factor for implicit rewards
        temperature: Temperature for loss computation
        intra_comparison_mode: "cumulative" for cumulative rewards, "stepwise" for single-step rewards
        
    Returns:
        preference_loss: Scalar tensor
        metrics: Dict with loss statistics
    """
    if not preference_data["pairs"]:
        # No preference pairs, return zero loss
        return torch.tensor(0.0, device=preference_data["device"]), {}
    
    # Compute implicit rewards: β * log(π_θ(y|x) / π_ref(y|x))
    implicit_rewards = beta * (log_probs - ref_log_probs)
    
    # Aggregate implicit rewards per trajectory (sum over response tokens)
    trajectory_rewards = (implicit_rewards * response_mask).sum(dim=-1)  # (batch_size,)
    
    # Group pairs by trajectory sets for Lambda weight computation
    if use_lambda_weights:
        # Separate intra and inter pairs for Lambda weight computation
        intra_groups = {}  # trajectory_idx -> list of pairs within that trajectory
        inter_groups = {}  # group_id -> list of pairs within that group
        
        for pair_idx, (better_traj, worse_traj, better_turn, worse_turn) in enumerate(preference_data["pairs"]):
            if better_turn == 0 and worse_turn == 0:
                # Inter-trajectory pair - group by group_id if available
                # For now, treat each inter pair independently (can be improved with group_ids)
                group_key = f"inter_{min(better_traj, worse_traj)}_{max(better_traj, worse_traj)}"
                if group_key not in inter_groups:
                    inter_groups[group_key] = []
                inter_groups[group_key].append(pair_idx)
            else:
                # Intra-trajectory pair
                traj_key = better_traj
                if traj_key not in intra_groups:
                    intra_groups[traj_key] = []
                intra_groups[traj_key].append(pair_idx)
    
    # 预计算intra-trajectory的累积implicit rewards以提高性能
    intra_cumulative_rewards = {}
    # 预计算每个轨迹内所有turns的ranks (用于Lambda权重)
    trajectory_ranks = {}
    
    if any(pair[2] != 0 or pair[3] != 0 for pair in preference_data["pairs"]):
        token_rewards = preference_data.get("token_level_rewards")
        if token_rewards is not None:
            for traj_idx in set(pair[0] for pair in preference_data["pairs"] if pair[2] != 0 or pair[3] != 0):
                turn_mask = token_rewards[traj_idx] != 0
                turn_positions = turn_mask.nonzero(as_tuple=True)[0]
                if len(turn_positions) > 0:
                    # 预计算每个turn位置的累积reward
                    cumulative_rewards = []
                    turn_implicit_rewards = []
                    for pos in turn_positions:
                        cum_reward = (implicit_rewards[traj_idx, :pos.item()+1] * response_mask[traj_idx, :pos.item()+1]).sum()
                        cumulative_rewards.append(cum_reward)
                        # 收集turn-level implicit rewards用于排名
                        if intra_comparison_mode == "stepwise":
                            # 对于stepwise，我们需要turn span的implicit reward总和
                            turn_start = turn_positions[len(turn_implicit_rewards) - 1].item() + 1 if len(turn_implicit_rewards) > 0 else 0
                            response_start = response_mask[traj_idx].nonzero(as_tuple=True)[0][0].item()
                            turn_start = max(turn_start, response_start)
                            turn_end = pos.item()
                            turn_reward = (implicit_rewards[traj_idx, turn_start:turn_end+1] * 
                                         response_mask[traj_idx, turn_start:turn_end+1]).sum()
                            turn_implicit_rewards.append(turn_reward.item())
                        else:
                            # 对于cumulative，直接使用累积值
                            turn_implicit_rewards.append(cum_reward.item())
                    
                    intra_cumulative_rewards[traj_idx] = cumulative_rewards
                    
                    # 计算基于implicit rewards的ranks (降序排列，rank 1 = 最高分)
                    if len(turn_implicit_rewards) > 1:
                        sorted_indices = sorted(range(len(turn_implicit_rewards)), 
                                              key=lambda i: turn_implicit_rewards[i], reverse=True)
                        ranks = [0] * len(turn_implicit_rewards)
                        for rank, idx in enumerate(sorted_indices):
                            ranks[idx] = rank + 1  # 1-indexed ranks
                        trajectory_ranks[traj_idx] = ranks
                    else:
                        trajectory_ranks[traj_idx] = [1]  # 单个turn的rank就是1
    
    total_loss = 0.0
    intra_loss = 0.0
    inter_loss = 0.0
    num_pairs = 0
    num_intra_pairs = 0
    num_inter_pairs = 0
    
    # Collect all scores and pairs for Lambda weight computation
    all_scores = []
    all_pair_info = []
    
    for pair_idx, (better_traj, worse_traj, better_turn, worse_turn) in enumerate(preference_data["pairs"]):
        if better_turn == 0 and worse_turn == 0:
            # Inter-trajectory comparison: use the same combined scoring as in pair construction
            better_cumulative = trajectory_rewards[better_traj]
            worse_cumulative = trajectory_rewards[worse_traj]
            
            # Get best molecule scores if available
            if best_molecule_scores is not None:
                better_molecule = best_molecule_scores[better_traj]
                worse_molecule = best_molecule_scores[worse_traj]
                
                # Apply same normalization and combination logic as in build_preference_pairs
                inter_weight_cumulative = preference_data.get("inter_weight_cumulative", 0.5)
                
                # Simple combination without normalization (since we're doing pairwise comparison)
                better_score = inter_weight_cumulative * better_cumulative + (1 - inter_weight_cumulative) * better_molecule
                worse_score = inter_weight_cumulative * worse_cumulative + (1 - inter_weight_cumulative) * worse_molecule
            else:
                # Fallback to cumulative rewards only
                better_score = better_cumulative
                worse_score = worse_cumulative
            
            is_inter = True
        else:
            # Intra-trajectory comparison: use turn-specific implicit rewards
            # Since better_traj == worse_traj for intra comparisons
            traj_idx = better_traj
            
            if intra_comparison_mode == "stepwise":
                # NEW: Stepwise comparison - compare single-step implicit rewards
                token_rewards = preference_data.get("token_level_rewards")
                if token_rewards is not None:
                    turn_mask = token_rewards[traj_idx] != 0
                    turn_positions = turn_mask.nonzero(as_tuple=True)[0]
                    
                    if better_turn < len(turn_positions) and worse_turn < len(turn_positions):
                        better_pos = turn_positions[better_turn].item()
                        worse_pos = turn_positions[worse_turn].item()
                        
                        # Compare the sum of implicit rewards for the entire turn span
                        # For stepwise comparison, we need to compute the turn boundaries
                        
                        # Find turn start positions (previous turn end + 1, or sequence start)
                        better_start = turn_positions[better_turn - 1].item() + 1 if better_turn > 0 else 0
                        worse_start = turn_positions[worse_turn - 1].item() + 1 if worse_turn > 0 else 0
                        
                        # Turn end positions are already known
                        better_end = better_pos
                        worse_end = worse_pos
                        
                        # Find actual response start within the trajectory (skip prompt tokens)
                        response_start = response_mask[traj_idx].nonzero(as_tuple=True)[0][0].item()
                        better_start = max(better_start, response_start)
                        worse_start = max(worse_start, response_start)
                        
                        # Sum implicit rewards for the entire turn span
                        better_score = (implicit_rewards[traj_idx, better_start:better_end+1] * 
                                      response_mask[traj_idx, better_start:better_end+1]).sum()
                        worse_score = (implicit_rewards[traj_idx, worse_start:worse_end+1] * 
                                     response_mask[traj_idx, worse_start:worse_end+1]).sum()
                    else:
                        # Fallback to trajectory rewards
                        better_score = trajectory_rewards[traj_idx]
                        worse_score = trajectory_rewards[traj_idx]
                else:
                    # Fallback to trajectory rewards
                    better_score = trajectory_rewards[traj_idx]
                    worse_score = trajectory_rewards[traj_idx]
            else:
                # ORIGINAL: Cumulative comparison - compare cumulative implicit rewards
                # 使用预计算的累积rewards
                if traj_idx in intra_cumulative_rewards and better_turn >= 0 and worse_turn >= 0:
                    cumulative_rewards = intra_cumulative_rewards[traj_idx]
                    if better_turn < len(cumulative_rewards) and worse_turn < len(cumulative_rewards):
                        better_score = cumulative_rewards[better_turn]
                        worse_score = cumulative_rewards[worse_turn]
                    else:
                        # Fallback
                        better_score = trajectory_rewards[traj_idx]
                        worse_score = trajectory_rewards[traj_idx]
                else:
                    # Fallback：原始计算方法
                    token_rewards = preference_data.get("token_level_rewards")
                    if token_rewards is not None:
                        turn_mask = token_rewards[traj_idx] != 0
                        turn_positions = turn_mask.nonzero(as_tuple=True)[0]
                        
                        if better_turn < len(turn_positions) and worse_turn < len(turn_positions):
                            better_pos = turn_positions[better_turn].item()
                            worse_pos = turn_positions[worse_turn].item()
                            
                            better_score = (implicit_rewards[traj_idx, :better_pos+1] * response_mask[traj_idx, :better_pos+1]).sum()
                            worse_score = (implicit_rewards[traj_idx, :worse_pos+1] * response_mask[traj_idx, :worse_pos+1]).sum()
                        else:
                            better_score = trajectory_rewards[traj_idx]
                            worse_score = trajectory_rewards[traj_idx]
                    else:
                        better_score = trajectory_rewards[traj_idx]
                        worse_score = trajectory_rewards[traj_idx]
            is_inter = False
        
        # Store scores for Lambda weight computation
        all_scores.append((better_score, worse_score))
        all_pair_info.append((pair_idx, is_inter, better_traj, worse_traj, better_turn, worse_turn))
        
        # LiPO-λ loss: log(1 + exp(-(s_better - s_worse) / temperature))
        score_diff = (better_score - worse_score) / temperature
        pair_loss = torch.log(1 + torch.exp(-score_diff))
        
        # Compute Lambda weight if enabled
        if use_lambda_weights:
            # Get label scores from preference_data
            if pair_idx < len(preference_data["scores"]):
                better_label, worse_label = preference_data["scores"][pair_idx]
            else:
                # Fallback: use implicit reward scores as labels
                better_label, worse_label = better_score.item(), worse_score.item()
            
            # Compute Lambda weight following LiPO-λ paper exactly
            # Use environment rewards as preference labels (ψi in the paper)
            # Gain function: G(ψ) = 2^ψ - 1  
            gain_better = 2 ** better_label - 1
            gain_worse = 2 ** worse_label - 1
            
            # For intra-trajectory pairs, use precomputed trajectory ranks
            if not is_inter:
                traj_idx = better_traj
                if traj_idx in trajectory_ranks and len(trajectory_ranks[traj_idx]) > max(better_turn, worse_turn):
                    # Use actual ranks within the complete trajectory
                    better_rank = trajectory_ranks[traj_idx][better_turn]
                    worse_rank = trajectory_ranks[traj_idx][worse_turn]
                else:
                    # Fallback to simple comparison
                    if better_score > worse_score:
                        better_rank, worse_rank = 1, 2
                    else:
                        better_rank, worse_rank = 2, 1
                    
                # Rank discount function: D(τ) = log(1 + τ)
                discount_better = math.log(1 + better_rank)
                discount_worse = math.log(1 + worse_rank)
                
                # Lambda weight: |Gi - Gj| * |1/D(τi) - 1/D(τj)|
                gain_diff = abs(gain_better - gain_worse)
                discount_diff = abs(1.0 / (discount_better + 1e-8) - 1.0 / (discount_worse + 1e-8))
                lambda_weight = gain_diff * discount_diff
                
                # Clamp to reasonable range
                lambda_weight = max(0.01, min(lambda_weight, 5.0))
            else:
                # For inter-trajectory, simplified weighting by reward difference
                lambda_weight = abs(better_label - worse_label) + 0.1
        else:
            lambda_weight = 1.0
        
        # Apply intra/inter specific weights
        if is_inter:
            weighted_loss = lambda_weight * inter_weight * pair_loss
            inter_loss += lambda_weight * pair_loss  # Track unweighted for metrics
            num_inter_pairs += 1
        else:
            weighted_loss = lambda_weight * intra_weight * pair_loss
            intra_loss += lambda_weight * pair_loss  # Track unweighted for metrics
            num_intra_pairs += 1
        
        total_loss += weighted_loss
        num_pairs += 1
        
    # Calculate average losses for each type FIRST
    if num_intra_pairs > 0:
        avg_intra_loss = (intra_loss / num_intra_pairs)
        weighted_avg_intra = avg_intra_loss * intra_weight
    else:
        avg_intra_loss = torch.tensor(0.0, device=preference_data["device"])
        weighted_avg_intra = torch.tensor(0.0, device=preference_data["device"])
    
    if num_inter_pairs > 0:
        avg_inter_loss = (inter_loss / num_inter_pairs)
        weighted_avg_inter = avg_inter_loss * inter_weight
    else:
        avg_inter_loss = torch.tensor(0.0, device=preference_data["device"])
        weighted_avg_inter = torch.tensor(0.0, device=preference_data["device"])
    
    # Combine weighted averages (not weighted sums)
    if num_intra_pairs > 0 and num_inter_pairs > 0:
        # Both types present: sum of weighted averages (do not divide by 2!)
        preference_loss = weighted_avg_intra + weighted_avg_inter
    elif num_intra_pairs > 0:
        # Only intra
        preference_loss = weighted_avg_intra
    elif num_inter_pairs > 0:
        # Only inter
        preference_loss = weighted_avg_inter
    else:
        # No pairs
        preference_loss = torch.tensor(0.0, device=preference_data["device"])
    
    # Calculate average pairs per trajectory for clearer monitoring
    batch_size = preference_data.get("batch_size", 1)  # Get batch size from preference_data
    avg_pairs_per_trajectory = num_pairs / max(1, batch_size)
    avg_intra_pairs_per_trajectory = num_intra_pairs / max(1, batch_size)
    
    metrics = {
        "preference_loss/loss": preference_loss.item(),
        "preference_loss/avg_pairs_per_trajectory": avg_pairs_per_trajectory,
        "preference_loss/avg_score_diff": sum([s[0] - s[1] for s in preference_data["scores"]]) / max(1, len(preference_data["scores"])),
        # Separate metrics for intra and inter
        "preference_loss/intra_loss": avg_intra_loss.item() if torch.is_tensor(avg_intra_loss) else 0.0,
        "preference_loss/inter_loss": avg_inter_loss.item() if torch.is_tensor(avg_inter_loss) else 0.0,
        "preference_loss/avg_intra_pairs_per_trajectory": avg_intra_pairs_per_trajectory,
        "preference_loss/num_inter_pairs": num_inter_pairs
    }
    
    return preference_loss, metrics


# set up unittest
if __name__ == "__main__":
    token_level_rewards = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    loss_mask = torch.ones(1, 10)
    advantages, returns = compute_bi_level_gae_advantage_return(token_level_rewards, values, loss_mask, 1, 1, 0.95)
    print(advantages)
    print(returns)