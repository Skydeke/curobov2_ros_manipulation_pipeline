/* Ported from MoveIt's Motion Planning panel -- see the header for provenance
 * and for the one deliberate deviation (radians only).
 *
 * Upstream: src/motion_planning_frame_joints_widget.cpp, lines 397-560,
 *   Copyright (c) 2019, CITEC, Bielefeld University (BSD-3).
 */

#include <isaac_ros_cumotion_rviz/progress_bar_delegate.hpp>

#include <algorithm>
#include <cmath>

#include <QAbstractItemView>
#include <QApplication>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOptionProgressBar>
#include <QLocale>

namespace isaac_ros_cumotion_rviz
{

namespace
{

/// The bar is drawn over this many steps rather than in the value's own units,
/// so the fill fraction is the same regardless of how wide the joint's range is.
constexpr int kBarSteps = 1000;

}  // namespace

ProgressBarDelegate::ProgressBarDelegate(int editable_column, QObject * parent)
: QStyledItemDelegate(parent), editable_column_(editable_column)
{
}

/// Draws the base item (selection, focus, the text columns) and then, for the
/// one editable column, overwrites it with the progress bar carrying the value as
/// its caption. Structure follows MoveIt's ProgressBarDelegate::paint.
void ProgressBarDelegate::paint(
  QPainter * painter, const QStyleOptionViewItem & option, const QModelIndex & index) const
{
  QStyle * style = option.widget ? option.widget->style() : QApplication::style();
  QStyleOptionViewItem style_option = option;
  initStyleOption(&style_option, index);

  if (index.column() == editable_column_) {
    const QVariant bounds = index.data(VariableBoundsRole);
    if (bounds.isValid()) {
      const QPointF limits = bounds.toPointF();
      // EditRole, not DisplayRole. MoveIt can use DisplayRole because both of its
      // roles are two views of one RobotState. A QTreeWidget has no such shared
      // source: a drag commits through model->setData(..., Qt::EditRole), which
      // QTreeWidgetItem stores verbatim, leaving DisplayRole stale. Reading
      // EditRole here is what makes the fill follow the drag -- reading
      // DisplayRole instead paints a bar that never moves.
      const double value = index.data(Qt::EditRole).toDouble();

      QStyleOptionProgressBar bar;
      bar.rect = option.rect;
      bar.minimum = 0;
      bar.maximum = kBarSteps;
      // Clamped: a stored target can sit outside the bounds the model later
      // reports (limits were widened or narrowed since it was set), and a fill
      // outside 0..1000 is drawn by some styles as nonsense.
      const double span = limits.y() - limits.x();
      const double fraction = (span > 0.0) ? (value - limits.x()) / span : 0.0;
      bar.progress = static_cast<int>(
        std::lround(std::clamp(fraction, 0.0, 1.0) * kBarSteps));
      // Formatted from EditRole rather than reused from the item's DisplayRole
      // text, for the same reason the fill is: mid-drag only EditRole is
      // updated, and a caption lagging behind its own bar is worse than none.
      bar.text = QLocale().toString(value, 'f', 4);
      bar.textAlignment = option.displayAlignment;
      bar.textVisible = true;
      style->drawControl(QStyle::CE_ProgressBar, &bar, painter);
      return;
    }
  }

  style->drawControl(QStyle::CE_ItemViewItem, &style_option, painter, option.widget);
}

/// Returns the drag editor for the editable column, or nullptr for every other
/// column. nullptr is the important half: it is what stops a stray double-click
/// or keypress on the joint-NAME column from opening a line-edit editor that
/// would write a renamed joint back into the model.
QWidget * ProgressBarDelegate::createEditor(
  QWidget * parent, const QStyleOptionViewItem & /* option */,
  const QModelIndex & index) const
{
  if (index.column() == editable_column_) {
    const QVariant bounds = index.data(VariableBoundsRole);
    if (bounds.isValid()) {
      const QPointF limits = bounds.toPointF();
      auto * editor = new ProgressBarEditor(parent, limits.x(), limits.y(), 4);
      connect(editor, &ProgressBarEditor::editingFinished, this,
              &ProgressBarDelegate::commitAndCloseEditor);
      // Write every intermediate position to the model while dragging, so the
      // rest of the tab sees the bar move in real time rather than only on
      // release -- that live feedback is the whole point of the control.
      connect(editor, &ProgressBarEditor::valueChanged, this, [this, index](double value) {
        if (auto * model = const_cast<QAbstractItemModel *>(index.model())) {
          model->setData(index, value, Qt::EditRole);
        }
      });
      return editor;
    }
  }
  return nullptr;  // not this column, or unbounded: no editor at all
}

void ProgressBarDelegate::commitAndCloseEditor()
{
  auto * editor = qobject_cast<ProgressBarEditor *>(sender());
  if (editor == nullptr) {
    return;
  }
  commitData(editor);
  closeEditor(editor);
  Q_EMIT valueEdited();
}

ProgressBarEventFilter::ProgressBarEventFilter(QAbstractItemView * view, int editable_column)
: QObject(view), editable_column_(editable_column)
{
}

bool ProgressBarEventFilter::eventFilter(QObject * /*target*/, QEvent * event)
{
  if (event->type() == QEvent::MouseButtonPress) {
    auto * view = qobject_cast<QAbstractItemView *>(parent());
    if (view != nullptr) {
      const QModelIndex index = view->indexAt(static_cast<QMouseEvent *>(event)->pos());
      // MoveIt gates on ItemIsEditable; gating on the column instead is stricter
      // and is what we want, because on a QTreeWidget the item-level flag cannot
      // be per-cell and would make the name column draggable too.
      if (index.isValid() && index.column() == editable_column_ &&
        index.data(ProgressBarDelegate::VariableBoundsRole).isValid())
      {
        view->setCurrentIndex(index);
        view->edit(index);
        return true;  // handled: the bar takes the press, and drags from it
      }
    }
  }
  return false;
}

ProgressBarEditor::ProgressBarEditor(QWidget * parent, double min, double max, int digits)
: QWidget(parent), value_(min), min_(min), max_(max), digits_(digits)
{
  // This object is built from inside a mouse-press handler, so the press that
  // created it is still held down. Grabbing here is what lets the drag continue
  // once the cursor leaves the cell -- without it the editor only tracks the
  // mouse while it happens to be over the bar.
  if (QApplication::mouseButtons() & Qt::LeftButton) {
    grabMouse();
  }
}

void ProgressBarEditor::paintEvent(QPaintEvent * /*event*/)
{
  QPainter painter(this);

  QStyleOptionProgressBar bar;
  bar.rect = rect();
  bar.palette = palette();
  bar.minimum = 0;
  bar.maximum = kBarSteps;
  const double span = max_ - min_;
  const double fraction = (span > 0.0) ? (value_ - min_) / span : 0.0;
  bar.progress = static_cast<int>(std::lround(std::clamp(fraction, 0.0, 1.0) * kBarSteps));
  bar.text = QLocale().toString(value_, 'f', digits_);
  bar.textAlignment = Qt::AlignRight;
  bar.textVisible = true;
  style()->drawControl(QStyle::CE_ProgressBar, &bar, &painter);
}

void ProgressBarEditor::mousePressEvent(QMouseEvent * event)
{
  if (event->button() == Qt::LeftButton) {
    mouseMoveEvent(event);  // a click alone should already land on the value
  }
}

void ProgressBarEditor::mouseMoveEvent(QMouseEvent * event)
{
  // Absolute positioning, not relative: the bar maps the whole cell width onto
  // the joint's whole range, so the value under the cursor is min + fraction.
  const double span = max_ - min_;
  const double fraction = (width() > 0) ? double(event->x()) / double(width()) : 0.0;
  const double v = std::clamp(min_ + fraction * span, min_, max_);
  if (value_ != v) {
    value_ = v;
    Q_EMIT valueChanged(v);
    update();
  }
  event->accept();
}

void ProgressBarEditor::mouseReleaseEvent(QMouseEvent * event)
{
  if (event->button() == Qt::LeftButton) {
    releaseMouse();
    event->accept();
    Q_EMIT editingFinished();
  }
}

}  // namespace isaac_ros_cumotion_rviz
