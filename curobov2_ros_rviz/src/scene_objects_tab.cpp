#include <curobov2_ros_rviz/scene_objects_tab.hpp>

// Complete Qt types, not <QtWidgets>. The tabs that need several of these
// classes list them individually so a missing include is a compile error naming
// the class, instead of a transitive include that only works by accident.
#include <QColor>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QSignalBlocker>
#include <QTimer>
#include <QToolButton>

namespace curobov2_ros_rviz
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

/// `QDoubleSpinBox::setValue` CLAMPS to the widget's range instead of refusing
/// the value, so restoring a recorded number the form cannot represent would
/// silently round it into something that reads like the object and is not.
/// Every numeric field on selection goes through here, which flags that case.
void putValue(QDoubleSpinBox * box, double value, bool * clamped)
{
  if (value < box->minimum() || value > box->maximum()) {
    *clamped = true;
  }
  box->setValue(value);
}

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
  using Request = curobov2_ros_interfaces::srv::AddObject_Request;
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

  get_scene_objects_client_ =
    node_->create_client<curobov2_ros_interfaces::srv::GetSceneObjects>(
    ns + "get_scene_objects");
  remove_all_client_ = node_->create_client<std_srvs::srv::Trigger>(ns + "remove_all_objects");
  detach_client_ = node_->create_client<std_srvs::srv::Trigger>(ns + "detach_object");
  remove_client_ = node_->create_client<curobov2_ros_interfaces::srv::RemoveObject>(
    ns + "remove_object");
  attach_client_ = node_->create_client<curobov2_ros_interfaces::srv::AttachObject>(
    ns + "attach_object");
  add_client_ =
    node_->create_client<curobov2_ros_interfaces::srv::AddObject>(ns + "add_object");

  scene_objects_.clear();
  // A different planner node is a different scene: drop what the old one said
  // rather than showing it against the new one for up to 2 s.
  known_.clear();
  // another node's scene: whatever THIS node had fetched says nothing about it.
  scene_objects_.clear();
  // `setText` emits `textChanged` and `setValue` emits `valueChanged`, neither of
  // which anything in this tab is connected to (the form is only ever READ when
  // a request is built), so clearing the widgets below cannot re-enter
  // showSelectedObject(). Asserted here because that is the property the whole
  // clear-then-repopulate relies on, and a future connect would break it
  // silently rather than loudly.
  ui_->object_name->clear();
  ui_->mesh_file_path->clear();
  ui_->object_x->setValue(0.0);
  ui_->object_y->setValue(0.0);
  ui_->object_z->setValue(0.0);
  ui_->object_rx->setValue(0.0);
  ui_->object_ry->setValue(0.0);
  ui_->object_rz->setValue(0.0);
  ui_->object_rw->setValue(1.0);
  ui_->shape_size_x_spin_box->setValue(0.0);
  ui_->shape_size_y_spin_box->setValue(0.0);
  ui_->shape_size_z_spin_box->setValue(0.0);
  ui_->color_r->setValue(0.0);
  ui_->color_g->setValue(0.0);
  ui_->color_b->setValue(0.0);
  ui_->color_a->setValue(1.0);

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
  // `showSelectedObject` owns the status line while a row is selected: it can
  // say more than just the name, and it is the one place that knows which of
  // the two cases applies.
  if (!selectedName().isEmpty()) {
    showSelectedObject();
  }
  setButtonsEnabled();
}

void SceneObjectsTab::showSelectedObject()
{
  const QString name = selectedName();
  if (name.isEmpty()) {
    return;
  }

  // The name goes in first and unconditionally: it is the one field that is
  // always knowable, because `get_scene_objects` reports names.
  ui_->object_name->setText(name);

  const auto it = scene_objects_.constFind(name);
  if (it == scene_objects_.constEnd()) {
    // An object in the list has no entry from the last reply. That means the
    // server's parallel arrays were short of a record, which is a protocol
    // violation, but is not recoverable on the GUI thread: the row still exists,
    // so show the name and explain the remainder could not be read.
    setStatus(
      tr("selected '%1' — server replied without geometry for this object")
        .arg(name));
    return;
  }

  // Every field below is read back out of the last `get_scene_objects` reply.
  const SceneObject & o = it.value();

  // `findData`, not an index cast: the combo's item order is filled in code
  // (curobo's own type constants), and the reply's type is the same constant,
  // so a reorder of that list cannot turn one shape into another.
  const int idx = ui_->shapes_combo_box->findData(o.type);
  if (idx >= 0) {
    ui_->shapes_combo_box->setCurrentIndex(idx);
  }
  ui_->mesh_file_path->setText(o.mesh_file_path);

  bool clamped = false;
  putValue(ui_->object_x, o.position[0], &clamped);
  putValue(ui_->object_y, o.position[1], &clamped);
  putValue(ui_->object_z, o.position[2], &clamped);
  putValue(ui_->object_rx, o.orientation[0], &clamped);
  putValue(ui_->object_ry, o.orientation[1], &clamped);
  putValue(ui_->object_rz, o.orientation[2], &clamped);
  putValue(ui_->object_rw, o.orientation[3], &clamped);
  putValue(ui_->shape_size_x_spin_box, o.dimensions[0], &clamped);
  putValue(ui_->shape_size_y_spin_box, o.dimensions[1], &clamped);
  putValue(ui_->shape_size_z_spin_box, o.dimensions[2], &clamped);
  putValue(ui_->color_r, o.color[0], &clamped);
  putValue(ui_->color_g, o.color[1], &clamped);
  putValue(ui_->color_b, o.color[2], &clamped);
  putValue(ui_->color_a, o.color[3], &clamped);

  if (clamped) {
    setStatus(
      tr("selected '%1' — some values are outside this form's range and were "
         "clamped; the server holds the truth")
        .arg(name));
  } else {
    setStatus(tr("selected '%1'%2")
                .arg(name, o.attached ? tr(" (attached to arm)") : QString()));
  }
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
    // The row's TEXT is decorated for an attached object, but the service call
    // is built from kNameRole, never from this text -- see kNameRole. So the
    // decoration cannot leak a " (attached)" suffix into an `attach_object` or
    // `remove_object` request, which is why it is safe to show at all.
    const bool attached =
      scene_objects_.value(n).attached;
    auto * item = new QListWidgetItem(
      attached ? tr("%1  [attached]").arg(n) : n, ui_->collision_objects_list);
    item->setData(kNameRole, n);
    if (attached) {
      // A second, non-textual signal of the same thing: colour is the part a
      // glance picks up, and it survives any future re-translation of the text.
      item->setForeground(QColor(Qt::darkYellow));
    }
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
  if (get_scene_objects_client_ == nullptr || !get_scene_objects_client_->service_is_ready()) {
    return;
  }
  if (refreshing_) {
    return;  // a query is already out; do not stack them
  }
  refreshing_ = true;
  setButtonsEnabled();
  auto request = std::make_shared<curobov2_ros_interfaces::srv::GetSceneObjects::Request>();
  get_scene_objects_client_->async_send_request(
    request,
    [this](rclcpp::Client<curobov2_ros_interfaces::srv::GetSceneObjects>::SharedFuture future) {
      QStringList names;
      bool success = false;
      QString message;
      QHash<QString, SceneObject> scene_objects;
      try {
        auto result = future.get();
        success = result->success;
        message = QString::fromStdString(result->message);

        const size_t n = result->names.size();
        // Build a map keyed by name in the order returned. If parallel arrays
        // are ever ragged (a logic error on the server), this tolerates it by
        // only filling for entries that have a name, but it cannot fix the rest.
        for (size_t i = 0; i < n; ++i) {
          const QString nm = QString::fromStdString(result->names[i]);
          if (nm.isEmpty()) {
            continue;  // should not happen; server must not return unnamed objects
          }
          SceneObject o;
          if (i < result->types.size()) {
            o.type = static_cast<int>(result->types[i]);
          }
          if (i < result->mesh_file_paths.size()) {
            o.mesh_file_path = QString::fromStdString(result->mesh_file_paths[i]);
          }
          if (i < result->poses.size()) {
            o.position[0] = result->poses[i].position.x;
            o.position[1] = result->poses[i].position.y;
            o.position[2] = result->poses[i].position.z;
            o.orientation[0] = result->poses[i].orientation.x;
            o.orientation[1] = result->poses[i].orientation.y;
            o.orientation[2] = result->poses[i].orientation.z;
            o.orientation[3] = result->poses[i].orientation.w;
          }
          if (i < result->dimensions.size()) {
            o.dimensions[0] = result->dimensions[i].x;
            o.dimensions[1] = result->dimensions[i].y;
            o.dimensions[2] = result->dimensions[i].z;
          }
          if (i < result->colors.size()) {
            o.color[0] = result->colors[i].r;
            o.color[1] = result->colors[i].g;
            o.color[2] = result->colors[i].b;
            o.color[3] = result->colors[i].a;
          }
          if (i < result->attached.size()) {
            o.attached = result->attached[i];
          }
          scene_objects.insert(nm, o);
          names.append(nm);
        }
        // Sorted so the list does not reshuffle whenever the server's bucket
        // walk order changes (it iterates cuboid, sphere, capsule, cylinder,
        // mesh), and so `names != known_` below means "the set of names really
        // changed" rather than "the same names in a new order".
        names.sort(Qt::CaseInsensitive);
      } catch (const std::exception & e) {
        message = QString::fromStdString(e.what());
      }
      runOnGuiThread([this, names, scene_objects, success, message]() {
        refreshing_ = false;
        if (!success) {
          setStatus(message.isEmpty() ? tr("get_scene_objects failed") : message);
          setButtonsEnabled();
          return;
        }
        // Always replaced, never merged: the reply is the whole truth about
        // this scene, so anything the previous reply said that this one does not
        // is gone. This is what keeps an object removed and re-added under the
        // same name between two polls from showing the FIRST one's geometry.
        scene_objects_ = scene_objects;

        // The rows only move when something VISIBLE changed -- the name set, or
        // which objects are attached, since that is drawn on the row. Rebuilding
        // on every 2 s poll would drop the selection and scroll position out from
        // under the operator even when nothing they can see has moved.
        //
        // `known_` therefore holds "name=attached" pairs, not bare names: a
        // signature is what makes "the attached set changed" detectable at all.
        QStringList signature;
        signature.reserve(names.size());
        for (const QString & n : names) {
          signature << (scene_objects.value(n).attached ? n + "=1" : n + "=0");
        }
        if (signature != known_) {
          known_ = signature;
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
  using Request = curobov2_ros_interfaces::srv::AddObject_Request;
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
    request,
    [this, name](
      rclcpp::Client<curobov2_ros_interfaces::srv::AddObject>::SharedFuture future) {
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
    auto request = std::make_shared<curobov2_ros_interfaces::srv::RemoveObject::Request>();
    request->name = name.toStdString();
    remove_client_->async_send_request(
      request, [this, name](
                 rclcpp::Client<curobov2_ros_interfaces::srv::RemoveObject>::SharedFuture
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
  auto request = std::make_shared<curobov2_ros_interfaces::srv::AttachObject::Request>();
  request->object_name = name.toStdString();
  attach_client_->async_send_request(
    request, [this, name](
               rclcpp::Client<curobov2_ros_interfaces::srv::AttachObject>::SharedFuture
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

}  // namespace curobov2_ros_rviz
