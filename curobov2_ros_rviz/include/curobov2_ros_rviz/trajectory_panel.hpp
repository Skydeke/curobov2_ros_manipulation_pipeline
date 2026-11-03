#pragma once

#ifndef CUROBOV2_ROS_RVIZ__TRAJECTORY_PANEL_HPP_
#define CUROBOV2_ROS_RVIZ__TRAJECTORY_PANEL_HPP_

#include <functional>

#include <QWidget>

class QLabel;
class QPushButton;
class QSlider;

namespace curobov2_ros_rviz
{

/// Playback scrubber for CuroboPlanningDisplay, docked next to the render
/// panel. Plain QWidget: rviz_common::WindowManagerInterface::addPane takes a
/// QWidget, so no panel plugin registration is needed.
class TrajectoryPanel : public QWidget
{
public:
  explicit TrajectoryPanel(QWidget * parent = nullptr);

  void setDuration(double seconds);
  void setPosition(double seconds);
  double position() const;

  bool isPaused() const;
  void setPaused(bool paused);

  std::function<void(double)> onScrub;
  std::function<void(bool)> onPauseToggled;

private:
  void updateLabel();

  QSlider * slider_;
  QLabel * label_;
  QPushButton * pause_button_;
  double duration_ = 0.0;
  bool updating_ = false;
  bool paused_ = false;
};

}  // namespace curobov2_ros_rviz

#endif  // CUROBOV2_ROS_RVIZ__TRAJECTORY_PANEL_HPP_
