import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import copy

# --- 0. Cấu hình trang Streamlit ---
st.set_page_config(layout="wide", page_title="Minh Họa Cây Quyết Định")
st.title("🌳 Minh Họa Từng Bước Thuật Toán Cây Quyết Định")
st.markdown("---")

# --- 1. Tạo bộ dữ liệu VỊT - GÀ ---
@st.cache_data
def get_data():
    np.random.seed(42)
    vit_mo_dai = np.random.uniform(low=2.8, high=4.5, size=(10, 1))
    vit_chan_da_dang = np.random.uniform(low=1.5, high=5.0, size=(10, 1))
    data_vit_1 = np.hstack((vit_mo_dai, vit_chan_da_dang))
    vit_mo_ngan_chan_dai = np.random.uniform(low=1.0, high=2.5, size=(5, 1))
    vit_chan_rat_dai = np.random.uniform(low=4.0, high=5.5, size=(5, 1))
    data_vit_2 = np.hstack((vit_mo_ngan_chan_dai, vit_chan_rat_dai))
    data_vit = np.vstack((data_vit_1, data_vit_2))
    labels_vit = np.zeros(data_vit.shape[0]) # 0 = Vịt

    ga_mo_ngan = np.random.uniform(low=0.8, high=2.7, size=(12, 1))
    ga_chan_ngan_vua = np.random.uniform(low=1.0, high=3.8, size=(12, 1))
    data_ga_1 = np.hstack((ga_mo_ngan, ga_chan_ngan_vua))
    ga_mo_rat_ngan = np.random.uniform(low=0.5, high=1.5, size=(4,1))
    ga_chan_rat_ngan = np.random.uniform(low=0.8, high=2.0, size=(4,1))
    data_ga_2 = np.hstack((ga_mo_rat_ngan, ga_chan_rat_ngan))
    data_ga = np.vstack((data_ga_1, data_ga_2))
    labels_ga = np.ones(data_ga.shape[0]) # 1 = Gà

    X_original = np.vstack((data_vit, data_ga))
    y_original = np.concatenate((labels_vit, labels_ga))
    return X_original, y_original

X_original, y_original = get_data()
feature_names = ['Độ dài mỏ (cm)', 'Chiều dài chân (cm)']
class_names = ['Vịt', 'Gà']
colors_plot = ['#FF0000', '#0000FF'] # Red for Vit (0), Blue for Ga (1)
cmap_custom = ListedColormap(colors_plot)


# --- 2. Huấn luyện mô hình ---
def on_slider_change_callback():
    st.session_state.max_depth_slider_changed_flag = True

MAX_DEPTH_DEMO = st.sidebar.slider("Chọn độ sâu tối đa cho cây demo:", 1, 5, 3, key="max_depth_slider", on_change=on_slider_change_callback)

@st.cache_resource
def train_tree(_X, _y, max_depth):
    dt_classifier = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=max_depth)
    dt_classifier.fit(_X, _y)
    return dt_classifier

# Re-train tree if max_depth changes or on first run
# This check ensures the cached resource updates correctly when MAX_DEPTH_DEMO changes
if 'trained_max_depth' not in st.session_state or st.session_state.trained_max_depth != MAX_DEPTH_DEMO:
    dt_classifier = train_tree(X_original, y_original, MAX_DEPTH_DEMO)
    st.session_state.trained_max_depth = MAX_DEPTH_DEMO # Store the depth it was trained with
    # Clear potentially outdated state if tree structure changes
    if 'current_nodes_to_process' in st.session_state:
        initialize_state_logic(force_rerun=True)
else:
    # Retrieve from cache if depth hasn't changed since last train_tree call
    dt_classifier = train_tree(X_original, y_original, MAX_DEPTH_DEMO)

tree_ = dt_classifier.tree_


# --- Helper Function for Gini Calculation String ---
def format_gini_calculation(node_value_array, total_samples):
    """Formats a string explaining the Gini calculation for a node."""
    if total_samples == 0:
        return " (Không có mẫu)"
    gini_calc_str = "`1 - Σ(p_i)^2 = 1 - ("
    terms = []
    for i, count in enumerate(node_value_array):
        proportion = count / total_samples
        # Showing counts/total makes the calculation clearer
        terms.append(f"({int(count)}/{total_samples})^2")
        # Or show proportions: terms.append(f"{proportion:.3f}^2")
    gini_calc_str += " + ".join(terms) + ")`"
    return gini_calc_str

# --- 3. Khởi tạo/Reset Session State ---
def initialize_state_logic(force_rerun=False):
    st.session_state.current_nodes_to_process = [(0, X_original.copy(), y_original.copy(), 0, None)]
    st.session_state.history = [] # Stores (plot_args, info_text, nodes_to_process_before, step_val_before, node_id_displayed_before)
    st.session_state.step_counter = 0
    st.session_state.finished_processing = False
    st.session_state.max_depth_slider_changed_flag = False # Reset flag after use
    st.session_state.current_plot_args_cache = None
    st.session_state.processed_nodes = [] # Keep track of processed nodes if needed for other features

    # Initial info text for Step 0
    node_id = 0
    node_value_array = tree_.value[node_id][0]
    total_samples = tree_.n_node_samples[node_id]
    gini_value = tree_.impurity[node_id]
    gini_calc_explanation = format_gini_calculation(node_value_array, total_samples)
    root_counts_str = ", ".join([f"{class_names[i]}: {int(node_value_array[i])}" for i in range(len(class_names))])

    st.session_state.current_info_text_cache = (
        f"**Bước {st.session_state.step_counter}: Xem xét Nút Gốc `0` (Độ sâu 0)**\n"
        f"- **Tổng số mẫu:** {total_samples} \n"
        f"- **Gini impurity:** `{gini_value:.3f}` {gini_calc_explanation}\n"
    )
    if gini_value == 0.0:
        st.session_state.current_info_text_cache += "- *Nút đã thuần khiết!*\n"

    st.session_state.current_info_text_cache += f"\n➡️ **Nhấn 'Bước Tiếp Theo' để bắt đầu xây dựng cây.**"


    # Initial plot args cache for Step 0
    root_feature_idx = None
    root_threshold = None
    # Show potential split line even at step 0 if root node isn't a leaf
    if tree_.children_left[0] != _tree.TREE_LEAF and 0 < MAX_DEPTH_DEMO:
        root_feature_idx = tree_.feature[0]
        root_threshold = tree_.threshold[0]

    st.session_state.current_plot_args_cache = {
        'X_node_data': X_original.copy(), 'y_node_data': y_original.copy(),
        'title': f"Bước 0: Dữ liệu gốc (Nút 0)",
        'parent_split_info': None,
        'feature_idx': root_feature_idx, # Show potential split
        'threshold': root_threshold     # Show potential split
    }
    st.session_state.current_node_id_being_displayed = 0

    if force_rerun:
        st.rerun()

# Initialize state on first run or if slider changed
if 'current_nodes_to_process' not in st.session_state or st.session_state.get('max_depth_slider_changed_flag', False):
    # Set flag to false *before* initializing to prevent loop if init fails
    if 'max_depth_slider_changed_flag' in st.session_state:
         st.session_state.max_depth_slider_changed_flag = False
    initialize_state_logic(force_rerun=True) # Force rerun on init/slider change

# --- 4. Hàm trợ giúp để vẽ ---

# plot_data_at_node_streamlit remains the same as your last version
def plot_data_at_node_streamlit(ax, X_node_data, y_node_data, title, feature_idx=None, threshold=None, parent_split_info=None):
    ax.clear()
    unique_classes = np.unique(y_node_data)
    for class_label in unique_classes:
        mask = y_node_data == class_label
        ax.scatter(X_node_data[mask, 0], X_node_data[mask, 1],
                   c=colors_plot[int(class_label)],
                   label=class_names[int(class_label)],
                   edgecolor='k', s=80, alpha=0.8)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    counts = np.bincount(y_node_data.astype(int), minlength=len(class_names))
    class_counts_str = ", ".join([f"{class_names[i]}: {counts[i]}" for i in range(len(class_names))])
    ax.set_title(f"{title}\n{class_counts_str}", fontsize=10)
    ax.grid(True)
    xlim = (X_original[:,0].min()-0.5, X_original[:,0].max()+0.5)
    ylim = (X_original[:,1].min()-0.5, X_original[:,1].max()+0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc='lower right', title="Loài")

    if parent_split_info:
        p_feature, p_threshold, p_direction = parent_split_info
        line_style, line_color, fill_alpha = ':', 'gray', 0.1
        if p_feature == 0:
            ax.axvline(p_threshold, color=line_color, linestyle=line_style, linewidth=1.5)
            if p_direction == 'left': ax.fill_betweenx(ylim, xlim[0], p_threshold, color=line_color, alpha=fill_alpha)
            else: ax.fill_betweenx(ylim, p_threshold, xlim[1], color=line_color, alpha=fill_alpha)
        else:
            ax.axhline(p_threshold, color=line_color, linestyle=line_style, linewidth=1.5)
            if p_direction == 'left': ax.fill_between(xlim, ylim[0], p_threshold, color=line_color, alpha=fill_alpha)
            else: ax.fill_between(xlim, p_threshold, ylim[1], color=line_color, alpha=fill_alpha)

    if feature_idx is not None and threshold is not None:
        line_style, line_color = '--', 'green'
        if feature_idx == 0:
            ax.axvline(threshold, color=line_color, linestyle=line_style, linewidth=2)
            ax.text(threshold + 0.05, ylim[1] - 0.1, f'{feature_names[feature_idx]}\n<= {threshold:.2f}?', color=line_color, rotation=0, va='top', ha='left', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, pad=0.1))
        else:
            ax.axhline(threshold, color=line_color, linestyle=line_style, linewidth=2)
            ax.text(xlim[1] - 0.1, threshold + 0.05, f'{feature_names[feature_idx]}\n<= {threshold:.2f}?', color=line_color, va='bottom', ha='right', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, pad=0.1))
    return plt.gcf()

# --- Hàm vẽ cây đã được sửa đổi ---
def create_full_tree_plot_figure(dt_classifier_obj, f_names, c_names):
    """Creates a Matplotlib Figure object containing the full decision tree plot."""
    # --- TĂNG CÁC GIÁ TRỊ Ở ĐÂY ---
    fig_tree, ax = plt.subplots(figsize=(8, 6)) # Tăng kích thước tổng thể (ví dụ: 35x20 inches)
    plot_tree(dt_classifier_obj,
              ax=ax,
              filled=True,
              feature_names=f_names,
              class_names=c_names,
              rounded=True,
              fontsize=14, # Tăng kích thước chữ (ví dụ: 14)
              impurity=True,
              proportion=False,
              node_ids=True,
              label='all'
             )
   
    # --- KẾT THÚC PHẦN TĂNG GIÁ TRỊ ---
    plt.tight_layout() # Cố gắng điều chỉnh bố cục
    return fig_tree
# --- HẾT HÀM SỬA ĐỔI ---

# --- 5. Giao diện và Logic chính ---
# Adjust column ratio to give more space for info/queue
col_main, col_sidebar = st.columns([2, 2]) # e.g., 40% for info, 60% for tree

with col_main:
    st.subheader("Thông Tin Bước Xây Dựng Cây")
    info_placeholder = st.empty()
    st.markdown("---")
    st.subheader("Biểu Đồ Dữ Liệu Tại Nút")
    plot_placeholder = st.empty()
    st.markdown("---")
    st.subheader("Hàng Đợi Các Nút Cần Xử Lý") # Added Header for Queue
    queue_placeholder = st.empty() # Placeholder for Queue info

with col_sidebar:
    st.subheader("Cây Quyết Định Hoàn Chỉnh")
    tree_plot_placeholder = st.empty()
    st.markdown("---")
    st.sidebar.markdown("### Điều Khiển")

# --- Button Logic ---
button_cols = st.sidebar.columns(2)
with button_cols[0]:
    prev_disabled = not st.session_state.history
    if st.button("⬅️ Bước Trước", key="prev_step_btn", disabled=prev_disabled, use_container_width=True):
        if st.session_state.history:
            # Pop state: (plot_args, info_text, nodes_to_process_before, step_val_before, node_id_displayed_before)
            (prev_plot_args, prev_info_text, prev_nodes_to_process,
             prev_step_val, prev_node_id_displayed) = st.session_state.history.pop()

            # Restore state
            st.session_state.current_plot_args_cache = prev_plot_args
            st.session_state.current_info_text_cache = prev_info_text
            st.session_state.current_nodes_to_process = prev_nodes_to_process
            st.session_state.step_counter = prev_step_val
            st.session_state.current_node_id_being_displayed = prev_node_id_displayed
            st.session_state.finished_processing = False

            # Rerun to update display
            st.rerun()

with button_cols[1]:
    next_disabled = st.session_state.finished_processing or not st.session_state.current_nodes_to_process
    if st.button("➡️ Bước Tiếp Theo", key="next_step_btn", type="primary", disabled=next_disabled, use_container_width=True):
        if st.session_state.current_nodes_to_process:
            # --- Save State BEFORE processing ---
            nodes_to_process_before_pop = copy.deepcopy(st.session_state.current_nodes_to_process)
            # No need to save processed_nodes for this display logic
            st.session_state.history.append((
                copy.deepcopy(st.session_state.current_plot_args_cache),
                st.session_state.current_info_text_cache,
                nodes_to_process_before_pop,
                st.session_state.step_counter,
                st.session_state.current_node_id_being_displayed
            ))

            # --- Process Node ---
            node_id, X_subset, y_subset, depth, parent_split_info_for_current_node = st.session_state.current_nodes_to_process.pop(0)
            st.session_state.step_counter += 1
            st.session_state.current_node_id_being_displayed = node_id

            if node_id not in st.session_state.processed_nodes:
                st.session_state.processed_nodes.append(node_id)

            # --- Gather Info for Current Node ---
            is_leaf_in_tree = (tree_.children_left[node_id] == _tree.TREE_LEAF)
            is_leaf_due_to_depth = (depth >= MAX_DEPTH_DEMO)
            is_leaf = is_leaf_in_tree or is_leaf_due_to_depth

            node_value = tree_.value[node_id][0]
            total_samples = tree_.n_node_samples[node_id]
            gini_value = tree_.impurity[node_id]
            predicted_class_idx = np.argmax(node_value)
            predicted_class_name = class_names[predicted_class_idx]
            gini_calc_explanation = format_gini_calculation(node_value, total_samples)
            node_counts_str = ", ".join([f"{class_names[i]}: {int(node_value[i])}" for i in range(len(class_names))])

            # --- Build Info Text ---
            current_title_plot = f"Bước {st.session_state.step_counter}: Xét Nút `{node_id}` (Sâu {depth})"
            current_info_text = f"**{current_title_plot}**\n"
            current_info_text += f"- **Tổng số mẫu:** {total_samples} \n"
            current_info_text += f"- **Gini impurity:** `{gini_value:.3f}` {gini_calc_explanation}\n"
            if gini_value == 0.0:
                 current_info_text += "- *Nút thuần khiết!*\n"


            # --- Prepare Plot Arguments ---
            plot_args_current = {
                'X_node_data': X_subset.copy(), 'y_node_data': y_subset.copy(),
                'title': current_title_plot.replace("Xét", "Dữ liệu tại"), # Adjust title for plot
                'parent_split_info': parent_split_info_for_current_node,
                'feature_idx': None, # Default for leaf
                'threshold': None  # Default for leaf
            }

            # --- Handle Leaf vs. Split ---
            if is_leaf:
                leaf_reason = "(Do độ sâu tối đa)" if is_leaf_due_to_depth and not is_leaf_in_tree else ""
                current_info_text += f"- **==> NÚT LÁ** {leaf_reason}. **Dự đoán:** `{predicted_class_name}`\n"
                # plot_args already set for leaf
            else:
                # Node is being split
                feature = tree_.feature[node_id]
                threshold = tree_.threshold[node_id]
                left_child_id = tree_.children_left[node_id]
                right_child_id = tree_.children_right[node_id]

                current_info_text += f"- **Quyết định chia:** `{feature_names[feature]}` <= `{threshold:.2f}` ?\n"
                plot_args_current.update({'feature_idx': feature, 'threshold': threshold}) # Update plot args for split line

                # Calculate child node data
                left_mask = X_subset[:, feature] <= threshold
                right_mask = X_subset[:, feature] > threshold
                X_left, y_left = X_subset[left_mask], y_subset[left_mask]
                X_right, y_right = X_subset[right_mask], y_subset[right_mask]

                # Get info for children to display in text and potentially add to queue
                if left_child_id != _tree.TREE_LEAF:
                    left_value = tree_.value[left_child_id][0]
                    left_samples = tree_.n_node_samples[left_child_id] # Samples *in child node* from training
                    left_gini = tree_.impurity[left_child_id]
                    left_counts_str = ", ".join([f"{cl}: {int(v)}" for cl, v in zip(class_names, left_value)])
                    current_info_text += f"  - **Nhánh TRÁI (True):** -> Nút `{left_child_id}` ({X_left.shape[0]} mẫu đi vào, Gini nút con: {left_gini:.3f}, Counts: [{left_counts_str}])\n"
                    # Add to queue if it's not a leaf by depth
                    if (depth + 1) < MAX_DEPTH_DEMO and X_left.shape[0] > 0:
                         st.session_state.current_nodes_to_process.append((left_child_id, X_left, y_left, depth + 1, (feature, threshold, 'left')))
                    elif (depth + 1) >= MAX_DEPTH_DEMO and left_child_id not in st.session_state.processed_nodes:
                        st.session_state.processed_nodes.append(left_child_id) # Mark as processed if leaf due to depth

                if right_child_id != _tree.TREE_LEAF:
                    right_value = tree_.value[right_child_id][0]
                    right_samples = tree_.n_node_samples[right_child_id]
                    right_gini = tree_.impurity[right_child_id]
                    right_counts_str = ", ".join([f"{cl}: {int(v)}" for cl, v in zip(class_names, right_value)])
                    current_info_text += f"  - **Nhánh PHẢI (False):** -> Nút `{right_child_id}` ({X_right.shape[0]} mẫu đi vào, Gini nút con: {right_gini:.3f}, Counts: [{right_counts_str}])\n"
                    # Add to queue if it's not a leaf by depth
                    if (depth + 1) < MAX_DEPTH_DEMO and X_right.shape[0] > 0:
                         st.session_state.current_nodes_to_process.append((right_child_id, X_right, y_right, depth + 1, (feature, threshold, 'right')))
                    elif (depth + 1) >= MAX_DEPTH_DEMO and right_child_id not in st.session_state.processed_nodes:
                        st.session_state.processed_nodes.append(right_child_id) # Mark as processed if leaf due to depth


            # --- Update Cache for next display cycle ---
            st.session_state.current_plot_args_cache = plot_args_current
            st.session_state.current_info_text_cache = current_info_text

        # --- Check Finish Condition ---
        if not st.session_state.current_nodes_to_process and not st.session_state.finished_processing:
            st.session_state.finished_processing = True
            # Append completion message to the *cached* info text
            st.session_state.current_info_text_cache += "\n\n🎉 **Hoàn thành xây dựng cây (đến độ sâu tối đa)!** 🎉"

        # --- Rerun ---
        st.rerun()


# --- Display Current State (runs on each interaction/rerun) ---

# 1. Display Info Text (always uses cache)
info_placeholder.markdown(st.session_state.current_info_text_cache, unsafe_allow_html=True) # Allow markdown formatting like backticks

# 2. Display Data Plot (uses cache)
if st.session_state.current_plot_args_cache is not None:
    fig_main_display, ax_main_display = plt.subplots(figsize=(7, 6)) # Adjust size if needed
    plot_data_at_node_streamlit(ax_main_display, **st.session_state.current_plot_args_cache)
    plot_placeholder.pyplot(fig_main_display)
    plt.close(fig_main_display)

# 3. Display Queue Info (calculated fresh each time)
queue_info_text = "" # Start fresh
if st.session_state.current_nodes_to_process:
    # Sort queue for consistent display (by depth, then node ID)
    sorted_queue = sorted(st.session_state.current_nodes_to_process, key=lambda x: (x[3], x[0]))
    queue_info_text += "**Các nút đang chờ xử lý (Sắp xếp theo độ sâu, ID):**\n"
    for q_node_id, q_X_sub, _, q_depth, _ in sorted_queue:
        q_samples = q_X_sub.shape[0]
        q_gini = tree_.impurity[q_node_id]
        queue_info_text += f"- **Nút `{q_node_id}`**: Sâu `{q_depth}`, `{q_samples}` mẫu, Gini `{q_gini:.3f}`\n"
elif st.session_state.finished_processing:
    queue_info_text += "_Hàng đợi trống (Đã hoàn thành xây dựng cây)_"
elif st.session_state.step_counter > 0: # Don't show empty queue at step 0 start
     queue_info_text += "_Hàng đợi trống_"
# Only display the queue text if there's something to show or it's finished
if queue_info_text:
    queue_placeholder.markdown(queue_info_text)
else:
     # Clear the placeholder if queue is conceptually empty at start
     queue_placeholder.empty()


# 4. Display Full Tree Plot (regenerated each time)
fig_tree_display = create_full_tree_plot_figure(
    dt_classifier,
    feature_names,
    class_names
)
tree_plot_placeholder.pyplot(fig_tree_display)
plt.close(fig_tree_display) # Close immediately


# --- Reset Button ---
if st.sidebar.button("🔄 Reset Demo", key="reset_all_btn", use_container_width=True):
    # Clear cache associated with the specific training function if needed
    # train_tree.clear() # Usually not necessary unless hitting specific caching issues
    initialize_state_logic(force_rerun=True)


st.sidebar.markdown("---")
st.sidebar.header("Chú Giải")
st.sidebar.markdown("""
**Thông Tin Bước (Trái - Trên):**
- **Số mẫu:** Tổng số mẫu tại nút đang xét và số lượng mỗi lớp [Vịt, Gà].
- **Gini impurity:** Độ đo mức độ "lẫn lộn" của các lớp tại nút.
    - Công thức: `1 - Σ(p_i)^2`, với `p_i` là tỷ lệ mẫu của lớp `i`.
    - Gini = 0.0 nghĩa là nút hoàn toàn thuần khiết (chỉ chứa 1 lớp).
- **Quyết định chia:** Nếu nút không phải lá, hiển thị đặc trưng và ngưỡng dùng để chia.
- **Nhánh TRÁI/PHẢI:** Thông tin về nút con kết quả, bao gồm số mẫu đi vào nhánh đó, Gini của nút con, và số lượng mẫu mỗi lớp tại nút con.

**Biểu Đồ Dữ Liệu (Trái - Giữa):**
- Hiển thị các điểm dữ liệu thuộc về nút đang xét.
- Đường nét đứt màu xanh lá: Đường phân chia được quyết định tại nút này (nếu có).
- Vùng tô xám nhạt: Không gian dữ liệu của nút cha (nếu có).

**Hàng Đợi (Trái - Dưới):**
- Liệt kê các nút đang chờ được xử lý theo thứ tự ưu tiên (thường là theo chiều rộng - Breadth-First).

**Cây Quyết Định (Phải):**
- Hiển thị cấu trúc cây *hoàn chỉnh* được huấn luyện với độ sâu tối đa đã chọn.
- **samples**: Tổng số mẫu tại nút (trong dữ liệu huấn luyện ban đầu).
- **value**: Số lượng mẫu mỗi lớp [Vịt, Gà] tại nút.
- **class**: Lớp dự đoán đa số tại nút.
- **gini**: Gini impurity của nút.
""")
st.sidebar.markdown("---")
st.sidebar.info("Điều chỉnh 'Độ sâu tối đa' hoặc nhấn '🔄 Reset Demo' sẽ bắt đầu lại.")