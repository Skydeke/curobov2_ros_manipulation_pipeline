#include "curobov2_ros_rviz/trajectory_panel.hpp"

#include <algorithm>
#include <cmath>

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>

namespace curobov2_ros_rviz
{

TrajectoryPanel::TrajectoryPanel(QWidget * parent)
: QWidget(parent)
{
  slider_ = new QSlider(Qt::Horizontal, this);
  slider_->setRange(0, 1000);
  slider_->setValue(0);
  slider_->setEnabled(false);

  label_ = new QLabel(this);
  label_->setMinimumWidth(90);

  pause_button_ = new QPushButton(tr("Pause"), this);
  pause_button_->setEnabled(false);

  auto * layout = new QHBoxLayout(this);
  layout->setContentsMargins(4, 2, 4, 2);
  layout->addWidget(new QLabel(tr("Trajectory:"), this));
  layout->addWidget(slider_, 1);
  layout->addWidget(label_);
  layout->addWidget(pause_button_);

  connect(slider_, &QSlider::valueChanged, this, [this](int value) {
    if (updating_) {
      return;
    }
    updateLabel();
    if (onScrub) {
      onScrub(duration_ * static_cast<double>(value) / 1000.0);
    }
  });
  connect(pause_button_, &QPushButton::clicked, this, [this]() {
    setPaused(!paused_);
    if (onPauseToggled) {
      onPauseToggled(paused_);
    }
  });
}

void TrajectoryPanel::setDuration(double seconds)
{
  duration_ = std::max(0.0, seconds);
  slider_->setEnabled(duration_ > 0.0);
  pause_button_->setEnabled(duration_ > 0.0);
  setPosition(0.0);
  updateLabel();
}

void TrajectoryPanel::setPosition(double seconds)
{
  if (duration_ <= 0.0) {
    return;
  }
  const double fraction = std::clamp(seconds / duration_, 0.0, 1.0);
  updating_ = true;
  slider_->setValue(static_cast<int>(std::lround(fraction * 1000.0)));
  updating_ = false;
  updateLabel();
}

double TrajectoryPanel::position() const
{
  return duration_ * static_cast<double>(slider_->value()) / 1000.0;
}

bool TrajectoryPanel::isPaused() const
{
  return paused_;
}

void TrajectoryPanel::setPaused(bool paused)
{
  paused_ = paused;
  pause_button_->setText(paused_ ? tr("Play") : tr("Pause"));
}

void TrajectoryPanel::updateLabel()
{
  label_->setText(
    QString("%1 / %2 s")
    .arg(position(), 0, 'f', 2)
    .arg(duration_, 0, 'f', 2));
}

}  // namespace curobov2_ros_rviz
