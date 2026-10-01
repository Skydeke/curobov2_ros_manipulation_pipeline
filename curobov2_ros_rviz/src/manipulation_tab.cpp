#include <curobov2_ros_rviz/manipulation_tab.hpp>

#include <QListWidgetItem>

namespace curobov2_ros_rviz
{

ManipulationTab::ManipulationTab(QWidget * parent)
: QWidget(parent), ui_{std::make_unique<Ui::ManipulationTabForm>()}
{
  // The whole layout, the three group boxes in MoveIt's own grid cells and
  // MoveIt's own titles come from manipulation_tab.ui, which is a transcription
  // of MoveIt's Manipulation tab. See that file, and this class's header, for
  // what is absent and why.
  ui_->setupUi(this);

  // MoveIt fills these from its perception pipeline. There is no such pipeline
  // here, so both lists get the one row that says so. A QListWidget that is
  // merely empty looks like something failed; a disabled, unselectable row
  // reads as the deliberate absence it is.
  explainEmptyList(
    ui_->detected_objects_list,
    tr("No object recognition: curobo publishes no perception action, and no "
       "service returns a detected object. MoveIt's [Detect] button is absent "
       "for the same reason."));
  explainEmptyList(
    ui_->support_surfaces_list,
    tr("No support surfaces: there is no table or plane detector on this "
       "server. MoveIt's [Pick] and [Place] buttons are absent for the same "
       "reason."));

  // The ROI group keeps MoveIt's two row labels and nothing else. There is no
  // recogniser for a region of interest to bound, and TrajectoryGoal carries no
  // ROI field, so the six spinboxes MoveIt puts beside these labels are absent.
  // The labels are kept because they are what identifies the group as MoveIt's
  // ROI, and an empty group box with a title reads correctly; two labels with
  // nothing after them reads as a layout that did not finish.
}

ManipulationTab::~ManipulationTab() {}

void ManipulationTab::explainEmptyList(QListWidget * list, const QString & why)
{
  if (list == nullptr) {
    return;
  }
  auto * item = new QListWidgetItem(why, list);
  // Unselectable so the list cannot offer a row to act on: there is nothing to
  // select, because there is nothing in it.
  item->setFlags(Qt::NoItemFlags);
  list->setDisabled(true);
}

}  // namespace curobov2_ros_rviz
