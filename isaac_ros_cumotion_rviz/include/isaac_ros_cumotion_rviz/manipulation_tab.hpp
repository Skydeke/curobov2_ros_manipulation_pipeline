#pragma once

#include <QListWidget>
#include <QWidget>

#include <memory>

#include <ui_manipulation_tab.h>

namespace isaac_ros_cumotion_rviz
{

/// The Manipulation tab — MoveIt's Manipulation tab, and nothing else.
///
/// MoveIt's Manipulation tab is an OBJECT PERCEPTION tab. Its three groups are
/// fed by `moveit_ros_perception`:
///
///     Detected Objects     a list filled by the ObjectRecognition action,
///                          which `&Detect` fires
///     Support Surfaces     a list filled by a table / plane detector, whose
///                          `&Pick` and `P&lace` put the selected surface into
///                          the request
///     ROI                  the region of the camera image recognition runs
///                          over: "Center (m):" and "Size (m):" spinboxes
///
/// It has no gripper control at all — a ParallelJawGripper is driven from
/// MoveIt's separate ACTUATOR panel, not from this tab.
///
/// curobo has no perception of any kind, so all three groups have nothing behind
/// them. The tab is therefore MoveIt's FRAME without MoveIt's content: the same
/// QGridLayout, the same two columns, the same three group boxes in the same
/// grid cells, MoveIt's own group titles, and two empty lists. `&Detect`,
/// `&Pick`, `P&lace` and the six ROI spinboxes are omitted rather than
/// greyed out, and each omission names the service that would be needed;
/// `manipulation_tab.ui` carries the transcription and the reasons.
///
/// The empty lists are not left blank, because a blank box reads as a bug. Each
/// gets one disabled row saying which service is missing, added in the
/// constructor so the `.ui` stays a literal transcription of MoveIt's.
///
/// WHAT WAS HERE BEFORE, AND WHY IT IS GONE
/// This tab used to be MoveIt's Joints table scoped to the gripper joints
/// (names containing finger / gripper / claw), with Open, Close, Reset, Plan
/// and Plan + send on it, a `get_joint_info` poll, a `/joint_states`
/// subscription and its own `generate_trajectory` client and
/// `SendTrajectory` action client.
///
/// It was removed for two reasons.
///
/// It was the wrong tab. A table of gripper joints with a progress bar in the
/// value column IS MoveIt's Joints tab, so this looked like a duplicate of the
/// tab next to it rather than like MoveIt's Manipulation tab.
///
/// It duplicated that tab. `finger_joint` is in the cspace like every other
/// joint — the kortex config's cspace is `joint_1..joint_7` plus
/// `finger_joint` — so the Joints tab already lists it, already spans its bar
/// over its own limits, and already offers Set from current / Zero / Plan /
/// Plan + send against it. Opening and closing the gripper was a two-button
/// shortcut for dragging one row of a table that is right there.
///
/// One thing the old tab had that the Joints tab does not is worth naming:
/// Open and Close commanded the UPPER and LOWER limits by number, because
/// nothing in the cspace says which way a finger travels. The Joints tab's bar
/// does not guess either — it spans the joint's real limits and lets the
/// operator put the target where they want it. The honest version of the
/// shortcut is the bar.
///
/// This class therefore has no ROS state at all: no node, no clients, no
/// planner node, nothing to initialize. It is a frame, and it is here so the
/// tab exists in the panel's five at all, with MoveIt's shape.
class ManipulationTab : public QWidget
{
  Q_OBJECT

public:
  explicit ManipulationTab(QWidget * parent = nullptr);
  ~ManipulationTab() override;

private:
  /// Puts the one disabled explanatory row into `list`, so an empty list reads
  /// as a deliberate absence rather than a widget that failed to populate.
  static void explainEmptyList(QListWidget * list, const QString & why);

  std::unique_ptr<Ui::ManipulationTabForm> ui_;
};

}  // namespace isaac_ros_cumotion_rviz
