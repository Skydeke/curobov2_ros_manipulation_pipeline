#pragma once

#include <rclcpp/rclcpp.hpp>

#include <isaac_ros_cumotion_interfaces/srv/add_object.hpp>
#include <isaac_ros_cumotion_interfaces/srv/attach_object.hpp>
#include <isaac_ros_cumotion_interfaces/srv/remove_object.hpp>
#include <rviz_common/config.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <QListWidgetItem>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <functional>
#include <memory>

#include <ui_scene_objects_tab.h>

namespace isaac_ros_cumotion_rviz
{

/// Scene-object management — MoveIt's "Scene Objects" tab, laid out as MoveIt
/// lays it out.
///
/// SHAPE, FIRST, BECAUSE THAT IS WHAT WAS WRONG
/// MoveIt's tab (`scene_collision_objects` in
/// `motion_planning_rviz_plugin_frame.ui`) is a QHBoxLayout of two
/// QVBoxLayout columns:
///
///   LEFT   Current Scene Objects        a QListWidget
///          Add/Remove scene object(s)   three size spinboxes, then the shape
///                                       combo with [Add] [Del] [Clr]
///   RIGHT  Change object pose/scale     [ABSENT — see the .ui for why]
///          Object status                one status line
///          Scene Geometry               [ABSENT — see the .ui for why]
///
/// This widget is that shape, widget for widget, and `scene_objects_tab.ui`
/// carries the transcription with the deviations named next to them. The two
/// right-column groups that have no service behind them are ABSENT rather than
/// greyed out: a disabled control is a promise, and there is nothing here that
/// could ever keep it.
///
/// The add form used to be a separate AddObjectsPanel widget stacked above the
/// list in a QSplitter. That stacking is not a thing MoveIt has, and it was the
/// main reason the tab did not look like MoveIt's. AddObjectsPanel is gone; its
/// form and its `add_object` call now live in MoveIt's "Add/Remove scene
/// object(s)" group, ahead of MoveIt's own two rows.
///
/// ONE LIST, AND IT IS THE SERVER'S
/// `collision_objects_list` is filled from `get_obstacles`, so it shows what
/// the PLANNER has, not what this panel believes it added. Anything the task
/// constructor, the grasp orchestrator or a `ros2 service call` put in the world
/// shows up here and can be removed from here. AddObjectsPanel used to keep a
/// private QListWidget that it appended to on every successful add and deleted
/// from on every successful remove; that list could not see another client's
/// objects and so could disagree with the server, and it is exactly the kind of
/// write-only mirror that makes a scene editor lie. It is gone.
///
/// Why ONE flat list and not MoveIt's Present / Attached / Disabled split:
/// MoveIt can split because the planning scene message carries a per-object
/// state field. curobo's `get_obstacles` answers with newline-joined NAMES and
/// nothing else — no state, no type, no pose. Worse for the split specifically:
/// `attach` does not remove the object from the obstacle list, because the
/// server keeps it registered in the solver scenes and drops it only from the
/// voxel rasterisation, so an attached object still comes back from
/// `get_obstacles`. There is also no `get_attached_object`: the attached name
/// lives in a Python attribute and reaches the outside world only as the text
/// of `detach`'s reply. A two-group list would have to guess, and a guess here
/// reads as fact. The attach and detach ACTIONS are on the button row because
/// those take no state argument and work fine.
///
/// It also publishes nothing. Every entry shown is something the SERVER reports.
class SceneObjectsTab : public QWidget
{
  Q_OBJECT

public:
  explicit SceneObjectsTab(QWidget * parent = nullptr);
  ~SceneObjectsTab() override;

  void setPlannerNode(const QString & planner_node);
  QString plannerNode() const { return planner_node_; }
  void initialize(rclcpp::Node::SharedPtr node);

  // Both take a config and persist nothing from it. The signatures are fixed by
  // rviz_common::Panel; see the definitions for why the names are dropped there.
  void save(rviz_common::Config config) const;
  void load(const rviz_common::Config & config);

private Q_SLOTS:
  void refreshObjects();
  void onSelectionChanged();
  void addObject();
  void removeSelected();
  void removeAll();
  void attachSelected();
  void detachObject();

private:
  void runOnGuiThread(std::function<void()> fn);
  void setStatus(const QString & text);
  void setButtonsEnabled();
  /// The first non-empty name among the selected rows, or an empty string.
  QString selectedName() const;
  /// Repopulates `collision_objects_list` from `names`, keeping the selection
  /// and the scroll position. Only called when the name set actually changed,
  /// so an ordinary 2 s poll does not disturb what the operator is doing.
  void rebuildList(const QStringList & names);

  std::unique_ptr<Ui::SceneObjectsTabForm> ui_;

  rclcpp::Node::SharedPtr node_;
  QString planner_node_ = QStringLiteral("curobo_server");
  bool initialized_ = false;

  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr get_obstacles_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr remove_all_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr detach_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::RemoveObject>::SharedPtr remove_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::AttachObject>::SharedPtr attach_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::AddObject>::SharedPtr add_client_;

  /// Names currently listed, so a refresh can tell "nothing changed" from
  /// "everything changed" and leave the list (and the selection) alone.
  QStringList known_;
  bool refreshing_ = false;
  /// Set while an add/remove/clear/attach/detach call is out, so the buttons
  /// grey out for the duration instead of letting a second one mutate the
  /// request members underneath a reply that has not landed yet.
  bool busy_ = false;
};

}  // namespace isaac_ros_cumotion_rviz
