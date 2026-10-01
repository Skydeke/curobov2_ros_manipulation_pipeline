/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2019, CITEC, Bielefeld University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of CITEC / Bielefeld University nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 *  THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 *  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 *  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 *  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 *  DAMAGE.
 *********************************************************************/

/* Ported from MoveIt's Motion Planning panel: the delegate, the drag-to-edit
 * editor and the event filter behind the Joints tab's draggable bars.
 *
 * Upstream:
 *   moveit_ros/visualization/motion_planning_rviz_plugin/
 *     include/moveit/motion_planning_rviz_plugin/motion_planning_frame_joints_widget.hpp
 *     src/motion_planning_frame_joints_widget.cpp
 *   Copyright (c) 2019, CITEC, Bielefeld University (BSD-3, above)
 *
 * WHY A PORT AND NOT A QSlider
 * MoveIt does not put a QSlider in the joint table either -- it paints the value
 * cell itself as a QStyle progress bar (CE_ProgressBar) whose fill is the joint's
 * position within its own limits, and swaps a drag-to-set editor in on click. A
 * QSlider cannot do this: it draws its own handle and groove, has an integer
 * value, and would need a scale factor to cover a joint whose bounds differ by
 * three orders of magnitude from the next. The progress bar also draws the value
 * as text on top of the fill, which is what makes MoveIt's joint table readable
 * at a glance. Reproducing the look therefore means reproducing the delegate.
 *
 * THE DEVIATIONS FROM MOVEIT, ALL OF THEM
 * MoveIt's widget reads a RobotState, where a joint's bounds, its live value and
 * its target are three views of one object. A QTreeWidgetItem has no such shared
 * source: each role is stored independently and nothing keeps them in step. Every
 * deviation below follows from that, or from what curobo's service reports.
 *
 *  1. RADIANS ONLY. MoveIt converts REVOLUTE joints to degrees on the way in and
 *     out, because its RobotModel knows each joint's type. GetJointInfo reports
 *     no joint type -- only cspace names and position limits -- so there is
 *     nothing here to switch on, and guessing would silently mis-scale half the
 *     joints. Every other number in this panel and in the server's API is radians,
 *     so this stays radians; a degrees view is a per-joint-type feature, not a
 *     display choice.
 *  2. EditRole, NOT DisplayRole. MoveIt's paint() reads `index.data()`, i.e.
 *     DisplayRole, because a drag commits through setData(..., Qt::EditRole) into
 *     a model whose DisplayRole is a projection of the same RobotState, so the
 *     two agree. On a QTreeWidgetItem they do not: the drag writes EditRole and
 *     leaves DisplayRole stale, so painting from DisplayRole gives a bar whose
 *     fill never moves. Both the fill and the caption are read from EditRole.
 *  3. createEditor() RETURNS nullptr off-column. MoveIt returns the base-class
 *     editor, which for a QTreeView is a line edit -- so a double-click on a
 *     joint NAME in MoveIt would offer to rename it. Qt::ItemIsEditable is an
 *     item-level flag and cannot confine editing to one column, so nullptr is what
 *     confines it, together with the column test in ProgressBarEventFilter.
 *  4. NO BAR WITHOUT BOUNDS. MoveIt falls back to a wide default range for a
 *     joint with no limit. GetJointInfo reports math.inf for those, so a bar
 *     spanning them is not drawable and dragging one would drag a joint into
 *     infinity. Such a cell stays plain text and is not editable.
 *  5. THE FILL IS CLAMPED, and the editor divides by its width only when that
 *     width is non-zero. Neither is reachable from MoveIt's data (its bounds
 *     always contain its value, and a laid-out cell has a width), so neither was
 *     worth reproducing the failure mode of.
 */

#ifndef ISAAC_ROS_CUMOTION_RVIZ__PROGRESS_BAR_DELEGATE_HPP_
#define ISAAC_ROS_CUMOTION_RVIZ__PROGRESS_BAR_DELEGATE_HPP_

#include <QStyledItemDelegate>
#include <QWidget>

namespace curobov2_ros_rviz
{

/// Paints a bounded numeric column as a progress bar with the value written on
/// top, and edits it by dragging. Only `editableColumn` is drawn this way and
/// only that column can be edited -- every other column falls through to the
/// base class, so the text columns of the same table stay plain text.
class ProgressBarDelegate : public QStyledItemDelegate
{
  Q_OBJECT

public:
  /// Item data roles this delegate reads, beyond the Qt standard ones.
  enum CustomRole
  {
    /// QPointF(min, max) for the cell. A cell without this role is not drawn as
    /// a bar and is not editable, which is how a joint the model reports no
    /// bound for is left as plain text instead of getting a fake 0..1 range.
    VariableBoundsRole = Qt::UserRole + 1,
  };

  explicit ProgressBarDelegate(int editable_column, QObject * parent = nullptr);

  void paint(QPainter * painter, const QStyleOptionViewItem & option,
            const QModelIndex & index) const override;

  QWidget * createEditor(QWidget * parent, const QStyleOptionViewItem & option,
                         const QModelIndex & index) const override;

Q_SIGNALS:
  /// Emitted after a drag finished and the new value was written to the model.
  /// Lets the owning tab refresh whatever it derives from the edited value
  /// (a status line, a dependent preview) without polling.
  void valueEdited();

private Q_SLOTS:
  void commitAndCloseEditor();

private:
  int editable_column_;
};

/// The drag target: paints the same bar as the delegate and sets the value from
/// the mouse x position, clamped to the cell's bounds. Grabs the mouse on
/// construction because it is created from inside a mouse-press handler, so the
/// drag survives leaving the cell.
class ProgressBarEditor : public QWidget
{
  Q_OBJECT

public:
  ProgressBarEditor(QWidget * parent, double min, double max, int digits);

  void setValue(double value) {value_ = value;}
  double getValue() const {return value_;}

Q_SIGNALS:
  void valueChanged(double value);
  void editingFinished();

protected:
  void paintEvent(QPaintEvent * event) override;
  void mousePressEvent(QMouseEvent * event) override;
  void mouseMoveEvent(QMouseEvent * event) override;
  void mouseReleaseEvent(QMouseEvent * event) override;

private:
  double value_;
  double min_;
  double max_;
  int digits_;
};

/// Opens the bar editor on the FIRST mouse press over an editable cell, instead
/// of waiting for a double-click. Without this the bar is a dead graphic until
/// the user happens to double-click it, which is not how MoveIt behaves.
class ProgressBarEventFilter : public QObject
{
  Q_OBJECT

public:
  ProgressBarEventFilter(QAbstractItemView * view, int editable_column);

protected:
  bool eventFilter(QObject * target, QEvent * event) override;

private:
  int editable_column_;
};

}  // namespace curobov2_ros_rviz

#endif  // ISAAC_ROS_CUMOTION_RVIZ__PROGRESS_BAR_DELEGATE_HPP_
