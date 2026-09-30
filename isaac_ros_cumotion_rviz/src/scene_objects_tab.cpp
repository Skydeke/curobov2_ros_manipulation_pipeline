#include <isaac_ros_cumotion_rviz/scene_objects_tab.hpp>

// Complete Qt types, not <QtWidgets>. The tabs that need several of these
// classes list them individually so a missing include is a compile error naming
// the class, instead of a transitive include that only works by accident.
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QSignalBlocker>
#include <QTimer>
#include <QToolButton>

namespace isaac_ros_cumotion_rviz
{

namespace
{

/// Item data role holding the object's NAME.
///
/// The list's display text is the name too, but going through a role keeps the
/// service call independent of the visible text, so re-formatting a row cannot
/// silently start building an `attach_object` request out of a decorated string.
///
/// NOTE the argument order, because it is the opposite of the joint tables' and
/// the reason this constant is not shared with them: `QListWidgetItem::setData`
/// is `(role, value)`, whereas `QTreeWidgetItem::setData` is
/// `(column, role, value)`.
constexpr int kNameRole = Qt::UserRole;

}  // namespace

SceneObjectsTab::SceneObjectsTab(QWidget * parent)
: QWidget(parent), ui_{std::make_unique<Ui::SceneObjectsTabForm>()}
{
  // The whole layout, group boxes, button texts, tooltips and spinbox steps all
  // come from scene_objects_tab.ui, which is a transcription of MoveIt's own
  // Scene Objects tab. See the header comment there for what is absent and why.
  ui_->setupUi(this);

  // curobo's own shape constants. Three of MoveIt's five labels are used
  // verbatim because they are the same shape (Box, Sphere, Cylinder); the two
  // that are not ("Mesh from URL", "Cone") are replaced by the two curobo has
  // and MoveIt's version of the tab does not ("Capsule", and the one mesh form
  // curobo's AddObject.srv takes, a file path). Spelling a shape differently
  // while meaning the same one -- the old "Cylindre" -- helps nobody.
  using Request = isaac_ros_cumotion_interfaces::srv::AddObject_Request;
  ui_->shapes_combo_box->addItem(tr("Box"), QVariant(Request::CUBOID));
  ui_->shapes_combo_box->addItem(tr("Sphere"), QVariant(Request::SPHERE));
  ui_->shapes_combo_box->addItem(tr("Cylinder"), QVariant(Request::CYLINDER));
  ui_->shapes_combo_box->addItem(tr("Capsule"), QVariant(Request::CAPSULE));
  ui_->shapes_combo_box->addItem(tr("Mesh from file"), QVariant(Request::MESH));

  connect(
    ui_->collision_objects_list, &QListWidget::itemSelectionChanged, this,
    &SceneObjectsTab::onSelectionChanged);
  connect(ui_->add_object_button, &QToolButton::clicked, this, &SceneObjectsTab::addObject);
  connect(ui_->remove_object_button, &QToolButton::clicked, this, &SceneObjectsTab::removeSelected);
  connect(ui_->clear_scene_button, &QToolButton::clicked, this, &SceneObjectsTab::removeAll);
  connect(ui_->attach_object_button, &QToolButton::clicked, this, &SceneObjectsTab::attachSelected);
  connect(ui_->detach_object_button, &QToolButton::clicked, this, &SceneObjectsTab::detachObject);

  // MoveIt's list has no useful minimumSizeHint either, and this tab is not in a
  // scroll area, so without a floor a short dock would squeeze the list away.
  ui_->collision_objects_list->setMinimumHeight(120);

  setButtonsEnabled();
}

SceneObjectsTab::~SceneObjectsTab() {}

void SceneObjectsTab::initialize(rclcpp::Node::SharedPtr node)
{
  if (initialized_ || node == nullptr) {
    return;
  }
  node_ = node;
  initialized_ = true;
  setPlannerNode(planner_node_);

  // Poll, so an object added by the task constructor, the grasp orchestrator or
  // a `ros2 service call` appears without the operator doing anything.
  auto * timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, &SceneObjectsTab::refreshObjects);
  timer->start(2000);
}

void SceneObjectsTab::setPlannerNode(const QString & planner_node)
{
  planner_node_ =
    planner_node.trimmed().isEmpty() ? QStringLiteral("curobo_server") : planner_node.trimmed();
  if (node_ == nullptr) {
    return;
  }
  const std::string ns = "/" + planner_node_.toStdString() + "/";

  get_obstacles_client_ = node_->create_client<std_srvs::srv::Trigger>(ns + "get_obstacles");
  remove_all_client_ = node_->create_client<std_srvs::srv::Trigger>(ns + "remove_all_objects");
  detach_client_ = node_->create_client<std_srvs::srv::Trigger>(ns + "detach_object");
  remove_client_ = node_->create_client<isaac_ros_cumotion_interfaces::srv::RemoveObject>(
    ns + "remove_object");
  attach_client_ = node_->create_client<isaac_ros_cumotion_interfaces::srv::AttachObject>(
    ns + "attach_object");
  add_client_ =
    node_->create_client<isaac_ros_cumotion_interfaces::srv::AddObject>(ns + "add_object");

  // A different planner node is a different scene: drop what the old one said
  // rather than showing it against the new one for up to 2 s.
  known_.clear();
  refreshing_ = false;
  busy_ = false;
  ui_->collision_objects_list->clear();
  refreshObjects();
  setStatus(tr("planner node: %1").arg(planner_node_));
}

void SceneObjectsTab::runOnGuiThread(std::function<void()> fn)
{
  QMetaObject::invokeMethod(this, std::move(fn), Qt::QueuedConnection);
}

void SceneObjectsTab::setStatus(const QString & text)
{
  ui_->object_status->setText(text);
}

void SceneObjectsTab::setButtonsEnabled()
{
  const bool ready = initialized_ && !refreshing_ && !busy_;
  const bool has_selection = !selectedName().isEmpty();
  ui_->add_object_button->setEnabled(ready);
  ui_->remove_object_button->setEnabled(ready && has_selection);
  ui_->clear_scene_button->setEnabled(ready);
  ui_->attach_object_button->setEnabled(ready && has_selection);
  ui_->detach_object_button->setEnabled(ready);
}

void SceneObjectsTab::onSelectionChanged()
{
  // MoveIt's `object_status` describes the SELECTED object, so the selection
  // owns this label whenever there is one; a reply from a service overwrites it
  // and the next selection change puts the object back.
  const QString name = selectedName();
  if (!name.isEmpty()) {
    setStatus(tr("selected '%1'").arg(name));
  }
  setButtonsEnabled();
}

QString SceneObjectsTab::selectedName() const
{
  const QList<QListWidgetItem *> selected = ui_->collision_objects_list->selectedItems();
  for (QListWidgetItem * item : selected) {
    // Defensive: every row carries a name now that the list is flat, but an
    // empty one must not reach the service as an empty-name request.
    const QString name = item->data(kNameRole).toString();
    if (!name.isEmpty()) {
      return name;
    }
  }
  return {};
}

void SceneObjectsTab::rebuildList(const QStringList & names)
{
  const QString keep = selectedName();

  // Block the selection signal while the rows are torn down and rebuilt, so
  // `clear()` does not fire onSelectionChanged() once per removed row and paint
  // a status line for an object that no longer exists.
  const QSignalBlocker blocker(ui_->collision_objects_list);
  ui_->collision_objects_list->clear();
  for (const QString & n : names) {
    auto * item = new QListWidgetItem(n, ui_->collision_objects_list);
    item->setData(kNameRole, n);
  }
  if (!keep.isEmpty() && names.contains(keep)) {
    for (int row = 0; row < ui_->collision_objects_list->count(); ++row) {
      QListWidgetItem * item = ui_->collision_objects_list->item(row);
      if (item->data(kNameRole).toString() == keep) {
        ui_->collision_objects_list->setCurrentRow(row);
        break;
      }
    }
  }
}

void SceneObjectsTab::refreshObjects()
{
  if (get_obstacles_client_ == nullptr || !get_obstacles_client_->service_is_ready()) {
    return;
  }
  if (refreshing_) {
    return;  // a query is already out; do not stack them
  }
  refreshing_ = true;
  setButtonsEnabled();
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  get_obstacles_client_->async_send_request(
    request, [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      QStringList names;
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        // get_obstacles answers with the object names joined by newlines.
        const QString payload = QString::fromStdString(result->message);
        names = payload.split('\n', Qt::SkipEmptyParts);
        for (QString & n : names) {
          n = n.trimmed();
        }
        names.removeAll(QString());
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread([this, names, success, message]() {
        refreshing_ = false;
        if (!success) {
          setStatus(message.isEmpty() ? tr("get_obstacles failed") : message);
          setButtonsEnabled();
          return;
        }
        if (names != known_) {
          known_ = names;
          rebuildList(names);
        }
        setStatus(
          names.isEmpty()
            ? tr("scene has no added objects")
            : tr("%1 object(s) in the planning scene").arg(names.size()));
        setButtonsEnabled();
      });
    });
}

void SceneObjectsTab::addObject()
{
  if (add_client_ == nullptr || !add_client_->service_is_ready()) {
    setStatus(tr("add_object service not available"));
    return;
  }

  // Read every field into locals BEFORE the request goes out. The reply arrives
  // asynchronously and the widgets can be edited again the moment the button
  // re-enables, so anything read inside the callback would be the operator's
  // NEXT edit rather than the one that was submitted.
  using Request = isaac_ros_cumotion_interfaces::srv::AddObject_Request;
  auto request = std::make_shared<Request>();

  request->type = ui_->shapes_combo_box->currentData().toInt();
  const QString name = ui_->object_name->text().trimmed();
  const QString mesh_file_path = ui_->mesh_file_path->text().trimmed();
  if (name.isEmpty()) {
    setStatus(tr("the object must have a name — the server keys objects by it"));
    return;
  }
  if (request->type == Request::MESH && mesh_file_path.isEmpty()) {
    setStatus(tr("a mesh object needs a mesh path"));
    return;
  }
  request->name = name.toStdString();
  request->mesh_file_path = mesh_file_path.toStdString();

  request->pose.position.x = ui_->object_x->value();
  request->pose.position.y = ui_->object_y->value();
  request->pose.position.z = ui_->object_z->value();
  request->pose.orientation.x = ui_->object_rx->value();
  request->pose.orientation.y = ui_->object_ry->value();
  request->pose.orientation.z = ui_->object_rz->value();
  request->pose.orientation.w = ui_->object_rw->value();
  // The three size spinboxes are MoveIt's own shape_size_x/y/z_spin_box; for a
  // mesh the server ignores `dimensions` and reads the file.
  request->dimensions.x = ui_->shape_size_x_spin_box->value();
  request->dimensions.y = ui_->shape_size_y_spin_box->value();
  request->dimensions.z = ui_->shape_size_z_spin_box->value();
  request->color.r = ui_->color_r->value();
  request->color.g = ui_->color_g->value();
  request->color.b = ui_->color_b->value();
  request->color.a = ui_->color_a->value();

  busy_ = true;
  setButtonsEnabled();
  add_client_->async_send_request(
    request, [this, name](rclcpp::Client<isaac_ros_cumotion_interfaces::srv::AddObject>::SharedFuture
                           future) {
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread([this, name, success, message]() {
        busy_ = false;
        setStatus(success ? tr("added '%1': %2").arg(name, message)
                          : tr("could not add '%1': %2").arg(name, message));
        // The list is the server's, so it is refreshed rather than appended to:
        // if the name was already taken the server said so, and if it accepted
        // then the object really is in the scene now.
        refreshObjects();
      });
    });
}

void SceneObjectsTab::removeSelected()
{
  if (remove_client_ == nullptr || !remove_client_->service_is_ready()) {
    setStatus(tr("remove_object service not available"));
    return;
  }

  // Snapshot the names first: the requests complete asynchronously and refresh
  // the list, so iterating over live items would read a list that is being
  // rebuilt underneath the loop.
  QStringList targets;
  for (QListWidgetItem * item : ui_->collision_objects_list->selectedItems()) {
    const QString name = item->data(kNameRole).toString();
    if (!name.isEmpty()) {
      targets << name;
    }
  }
  if (targets.isEmpty()) {
    return;
  }

  busy_ = true;
  setButtonsEnabled();
  for (const QString & name : targets) {
    auto request = std::make_shared<isaac_ros_cumotion_interfaces::srv::RemoveObject::Request>();
    request->name = name.toStdString();
    remove_client_->async_send_request(
      request, [this, name](
                 rclcpp::Client<isaac_ros_cumotion_interfaces::srv::RemoveObject>::SharedFuture
                   future) {
        bool success = false;
        QString message;
        try {
          auto result = future.get();
          success = result->success;
          message = QString::fromStdString(result->message);
        } catch (const std::exception & e) {
          message = QString::fromStdString(e.what());
        }
        runOnGuiThread([this, name, success, message]() {
          setStatus(success ? tr("removed '%1'").arg(name)
                            : tr("could not remove '%1': %2").arg(name, message));
          refreshObjects();
        });
      });
  }
}

void SceneObjectsTab::removeAll()
{
  if (remove_all_client_ == nullptr || !remove_all_client_->service_is_ready()) {
    setStatus(tr("remove_all_objects service not available"));
    return;
  }
  busy_ = true;
  setButtonsEnabled();
  setStatus(tr("removing all objects..."));
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  remove_all_client_->async_send_request(
    request, [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread([this, success, message]() {
        busy_ = false;
        setStatus(success ? tr("scene cleared: %1").arg(message)
                          : tr("could not clear the scene: %1").arg(message));
        refreshObjects();
      });
    });
}

void SceneObjectsTab::attachSelected()
{
  const QString name = selectedName();
  if (name.isEmpty()) {
    return;
  }
  if (attach_client_ == nullptr || !attach_client_->service_is_ready()) {
    setStatus(tr("attach_object service not available"));
    return;
  }
  busy_ = true;
  setButtonsEnabled();
  setStatus(
    tr("attaching '%1' at the current configuration — the server rebuilds its "
       "solvers (blocking, ~20 s)...")
      .arg(name));
  auto request = std::make_shared<isaac_ros_cumotion_interfaces::srv::AttachObject::Request>();
  request->object_name = name.toStdString();
  attach_client_->async_send_request(
    request, [this, name](
               rclcpp::Client<isaac_ros_cumotion_interfaces::srv::AttachObject>::SharedFuture
                 future) {
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread([this, name, success, message]() {
        busy_ = false;
        setStatus(success ? tr("attached '%1': %2").arg(name, message)
                          : tr("could not attach '%1': %2").arg(name, message));
        refreshObjects();
      });
    });
}

void SceneObjectsTab::detachObject()
{
  if (detach_client_ == nullptr || !detach_client_->service_is_ready()) {
    setStatus(tr("detach_object service not available"));
    return;
  }
  busy_ = true;
  setButtonsEnabled();
  setStatus(tr("detaching..."));
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  detach_client_->async_send_request(
    request, [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      bool success = false;
      QString message;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread([this, success, message]() {
        busy_ = false;
        setStatus(success ? tr("detached: %1").arg(message)
                          : tr("could not detach: %1").arg(message));
        refreshObjects();
      });
    });
}

// The parameter is left unnamed in the definitions on purpose: the package builds
// with -Wall -Wextra, and an unused parameter that is merely commented about is
// still -Wunused-parameter. The signature is fixed by the base class.
void SceneObjectsTab::save(rviz_common::Config) const
{
  // planner_node_name is the Context tab's to save -- see JointsTab::save().
  //
  // Nothing about the form is persisted either, deliberately: the shape, the
  // size, the name and the colour of a scene object are properties of the
  // PLANNER's scene, not of this panel, and this panel is not the scene's owner.
  // Writing them into the rviz config would be writing down a copy of a world
  // that another client can change at any moment, and reading them back would
  // then put those stale numbers next to a list that has moved on.
}

void SceneObjectsTab::load(const rviz_common::Config &)
{
  // No planner_node_name read here either -- see JointsTab::save().
}

}  // namespace isaac_ros_cumotion_rviz
