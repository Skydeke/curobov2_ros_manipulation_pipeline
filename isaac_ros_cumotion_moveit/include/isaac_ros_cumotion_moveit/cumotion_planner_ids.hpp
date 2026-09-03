// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_CUMOTION_PLANNER_IDS_H
#define ISAAC_ROS_CUMOTION_PLANNER_IDS_H

#include <cstdint>
#include <string>
#include <vector>

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

/**
 * MoveIt planner IDs exposed through the planning pipeline. Each string maps to
 * a `SetPlanner.srv` planner type so the user can pick a specific cuRobo
 * planner through the MoveIt API (e.g. the RViz "Planner" dropdown).
 *
 * Only the Joint Space planner is exposed: it is the only one that honours a
 * MoveIt joint-space goal (`target_joint_positions`). The pose planners
 * (Classic/Multipoint) consume a Cartesian `target_pose`/`target_poses` and are
 * hidden to avoid silently planning toward an empty goal when a joint goal is
 * sent. Closed-loop planners (MPC, retargeting) are excluded too.
 *
 * `kAutoPlannerId` selects the Joint Space planner (the only available one).
 */
inline constexpr char kAutoPlannerId[] = "cuMotion";
inline constexpr char kJointSpacePlannerId[] = "cuMotion/JointSpace";

/// All MoveIt-facing planner IDs (including the auto/default one).
inline const std::vector<std::string> & plannerIds()
{
  static const std::vector<std::string> ids = {
    kAutoPlannerId,
    kJointSpacePlannerId,
  };
  return ids;
}

/**
 * Map a MoveIt planner ID to a `SetPlanner.srv` planner type constant.
 *
 * @param planner_id  The MoveIt planner ID from the request.
 * @param out_type    Receives the SetPlanner planner type when the ID names a
 *                    concrete cuRobo planner.
 * @return whether @p planner_id is recognized: true for `kAutoPlannerId`
 *         (auto, `out_type` left unchanged) or the Joint Space planner ID
 *         (`out_type` is set); false for an unknown ID.
 */
inline bool plannerIdToType(const std::string & planner_id, uint8_t & out_type)
{
  if (planner_id.empty() || planner_id == kAutoPlannerId) {
    return true;  // auto: the caller uses the Joint Space planner
  }
  if (planner_id == kJointSpacePlannerId) {
    out_type = 5;  // SetPlanner::Request::JOINT_SPACE
  } else {
    return false;
  }
  return true;
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_PLANNER_IDS_H
