#pragma once

#include <rclcpp/rclcpp.hpp>

#include <isaac_ros_cumotion_interfaces/srv/add_object.hpp>
#include <isaac_ros_cumotion_interfaces/srv/attach_object.hpp>
#include <isaac_ros_cumotion_interfaces/srv/get_scene_objects.hpp>
#include <isaac_ros_cumotion_interfaces/srv/remove_object.hpp>
#include <rviz_common/config.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <QHash>
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
/// `collision_objects_list` is filled from `get_scene_objects`, so it shows what
/// the PLANNER has, not what this panel believes it added. Anything the task
/// constructor, the grasp orchestrator or a `ros2 service call` put in the world
/// shows up here, can be removed from here, and — because the reply carries the
/// object's type, pose, size and colour — can have those shown on click too.
/// AddObjectsPanel used to keep a private QListWidget that it appended to on
/// every successful add and deleted from on every successful remove; that list
/// could not see another client's objects and so could disagree with the server,
/// and it is exactly the kind of write-only mirror that makes a scene editor
/// lie. It is gone.
///
/// Why ONE flat list and not MoveIt's Present / Attached / Disabled split:
/// `get_scene_objects` DOES report a per-object `attached` flag, which is what
/// used to make the split impossible — the attached name reached the outside
/// world only as the text of `detach`'s reply. So the split is now possible.
/// It is still not done, because MoveIt's third group ("Disabled") describes
/// per-object collision-enable state that curobo has no service for and no way
/// to read: curobo's `enable` flag lives in GPU buffers and is reset by any
/// world push. A group that is empty by construction is worse than no group.
/// Two groups would work, and an attached object's row is marked in the list
/// and named in `object_status` so the distinction is visible without them.
/// The attach and detach ACTIONS are on the button row because those take no
/// state argument and work fine.
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
  /// and the scroll position, and marks attached rows from `scene_objects_`.
  /// Only called when something visible changed (see `known_`), so an ordinary
  /// 2 s poll does not disturb what the operator is doing.
  void rebuildList(const QStringList & names);
  /// Copies the selected object's server-reported values into the widgets.
  /// Called from `onSelectionChanged()`.
  void showSelectedObject();

  /// One scene object exactly as `get_scene_objects` REPORTS it.
  ///
  /// There is deliberately no client-side record of what this panel sent. There
  /// used to be one, because the server could not be asked: `get_obstacles` is a
  /// bare `std_srvs/Trigger` whose payload is newline-joined NAMES and nothing
  /// else, so pose, size, colour and shape simply were not on the wire. That
  /// record was wrong in the ways a mirror of your own writes always is — an
  /// object placed by any other client had only a name, and an object removed
  /// and re-added under the same name showed the FIRST one's geometry.
  /// `get_scene_objects` answers from the scene the SOLVER holds, so this
  /// struct is replaced wholesale on every poll and cannot drift from it.
  ///
  /// It is read-only state: nothing here is used to build a request. curobo has
  /// no edit-object service, so editing the form after a click and pressing Add
  /// either re-adds under a new name or is rejected for reusing the old one.
  struct SceneObject
  {
    /// A `GetSceneObjects` type constant (CUBOID, SPHERE, ...), which are
    /// `AddObject`'s values.
    int type = 0;
    QString mesh_file_path;
    double position[3] = {0.0, 0.0, 0.0};
    /// Quaternion x, y, z, w.
    double orientation[4] = {0.0, 0.0, 0.0, 1.0};
    /// AddObject's `dimensions` convention. Which components are MEANINGFUL
    /// depends on `type`; see GetSceneObjects.srv.
    double dimensions[3] = {0.0, 0.0, 0.0};
    /// r, g, b, a.
    double color[4] = {0.0, 0.0, 0.0, 1.0};
    /// The object currently attached to the arm's `attached_object` link.
    bool attached = false;
  };

  std::unique_ptr<Ui::SceneObjectsTabForm> ui_;

  rclcpp::Node::SharedPtr node_;
  QString planner_node_ = QStringLiteral("curobo_server");
  bool initialized_ = false;

  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::GetSceneObjects>::SharedPtr
    get_scene_objects_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr remove_all_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr detach_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::RemoveObject>::SharedPtr remove_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::AttachObject>::SharedPtr attach_client_;
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::AddObject>::SharedPtr add_client_;

  /// A "name=attached" signature of what the list currently shows, one entry per
  /// row in display order. Compared against the next reply's equivalent to decide
  /// whether anything VISIBLE changed; when it has not, the list, the selection
  /// and the scroll position are all left alone. Attachment is part of the
  /// signature because it is drawn on the row, so a detach has to rebuild it.
  QStringList known_;
  bool refreshing_ = false;
  /// Every object the last successful `get_scene_objects` reply described, keyed
  /// by name. Replaced wholesale on every reply — never merged into, never
  /// pruned by hand — so it cannot describe an object the server no longer has.
  /// A name in the list with no entry here means the reply's parallel arrays
  /// were ragged (see `showSelectedObject`); the row still lists, with its name
  /// only.
  QHash<QString, SceneObject> scene_objects_;
  /// Set while an add/remove/clear/attach/detach call is out, so the buttons
  /// grey out for the duration instead of letting a second one mutate the
  /// request members underneath a reply that has not landed yet.
  bool busy_ = false;
};

}  // namespace isaac_ros_cumotion_rviz
