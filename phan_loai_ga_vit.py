# === Bước 1: Chuẩn bị đồ chơi ===
# Mình cần mấy món đồ chơi này để làm việc

import numpy as np # Đồ chơi xử lý số liệu (như bảng tính Excel mini)
import matplotlib.pyplot as plt # Đồ chơi để vẽ vời đồ thị, hình ảnh
from sklearn.tree import DecisionTreeClassifier, plot_tree # Đồ chơi chính: Máy học đoán Gà/Vịt và vẽ cái cây quyết định của nó
from sklearn.metrics import accuracy_score, classification_report # Đồ chơi để xem máy đoán đúng bao nhiêu %

# === Bước 2: Tự chế dữ liệu Gà và Vịt ===
# Tưởng tượng mình đo mỏ và chân của 15 con Vịt, 15 con Gà

# Đặt tên cho dễ nhớ
ten_dac_trung = ['Độ dài mỏ (cm)', 'Chiều dài chân (cm)'] # Mình sẽ đo 2 cái này
ten_loai = ['Vịt', 'Gà'] # Có 2 loại là Vịt (số 0) và Gà (số 1)

# ---- Dữ liệu Vịt (Gán nhãn là số 0) ----
np.random.seed(42) # Để lần nào chạy số liệu cũng giống nhau, dễ kiểm tra
mo_vit = np.random.uniform(3.0, 5.0, 15)  # Đo mỏ 15 con vịt (từ 3 đến 5 cm)
chan_vit = np.random.uniform(1.5, 3.5, 15) # Đo chân 15 con vịt (từ 1.5 đến 3.5 cm)
so_lieu_vit = np.column_stack((mo_vit, chan_vit)) # Ghép số đo mỏ và chân vịt thành bảng
nhan_vit = np.zeros(15, dtype=int) # Tạo 15 nhãn số 0 (0 là Vịt)

# ---- Dữ liệu Gà (Gán nhãn là số 1) ----
mo_ga = np.random.uniform(1.0, 3.2, 15)  # Đo mỏ 15 con gà (từ 1 đến 3.2 cm - hơi ngắn hơn vịt)
chan_ga = np.random.uniform(2.5, 4.5, 15) # Đo chân 15 con gà (từ 2.5 đến 4.5 cm - hơi dài hơn vịt)
so_lieu_ga = np.column_stack((mo_ga, chan_ga)) # Ghép số đo mỏ và chân gà thành bảng
nhan_ga = np.ones(15, dtype=int) # Tạo 15 nhãn số 1 (1 là Gà)

# ---- Gom tất cả lại ----
X = np.vstack((so_lieu_vit, so_lieu_ga)) # Chồng bảng vịt lên bảng gà -> có bảng số liệu 30 con
y = np.concatenate((nhan_vit, nhan_ga)) # Nối nhãn vịt và gà -> có danh sách 30 nhãn (0 hoặc 1)

print("--- Xong phần chuẩn bị dữ liệu ---")
print("Tổng cộng có:", X.shape[0], "con vật") # In ra tổng số con vật
print("Mỗi con có:", X.shape[1], "số đo (đặc trưng)") # In ra số lượng số đo
print("Đây là số đo của 5 con đầu tiên:\n", X[:5]) # In thử 5 dòng đầu của bảng số liệu
print("Đây là nhãn của 5 con đó (0=Vịt, 1=Gà):", y[:5]) # In nhãn tương ứng
print("-" * 30) # In dòng gạch ngang cho đẹp

# === Bước 3: Nhìn thử dữ liệu (Vẽ hình) - Không bắt buộc nhưng hay ===
plt.figure(figsize=(8, 6)) # Tạo khung ảnh kích thước 8x6
# Vẽ chấm đỏ cho Vịt
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Vịt (0)', marker='o')
# Vẽ dấu X xanh cho Gà
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Gà (1)', marker='x')
plt.xlabel(ten_dac_trung[0]) # Ghi tên trục X (Mỏ)
plt.ylabel(ten_dac_trung[1]) # Ghi tên trục Y (Chân)
plt.title('Hình ảnh dữ liệu Gà và Vịt') # Tiêu đề hình
plt.legend() # Hiện chú thích (Đỏ là Vịt, Xanh là Gà)
plt.grid(True) # Vẽ lưới cho dễ nhìn
plt.show() # Hiển thị hình vẽ lên

# === Bước 4: Tạo "Bộ Não" Cây Quyết Định và Dạy Nó ===

# Tạo một "bộ não" cây quyết định còn trống
# Mình bảo nó đừng tạo cây phức tạp quá (max_depth=2), chỉ 2 tầng thôi cho dễ hiểu
# criterion='gini' là cách nó chọn câu hỏi hay nhất để hỏi (ví dụ: "Mỏ dài hơn 3cm không?")
may_hoc = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)

print("--- Bắt đầu dạy máy học ---")
# Dạy máy học bằng dữ liệu mình vừa tạo (X là số đo, y là nhãn Gà/Vịt)
# Lệnh .fit() này là máy tự nhìn dữ liệu và xây cái cây quyết định
may_hoc.fit(X, y)
print("Dạy xong!")
print("-" * 30)

# === Bước 5: Xem "Bộ Não" (Cây Quyết Định) Nó Học Được Gì ===
print("--- Đây là cái cây máy học đã học được ---")
plt.figure(figsize=(12, 8)) # Tạo khung ảnh to hơn để vẽ cây
# Vẽ cái cây ra
plot_tree(may_hoc, # Cái cây cần vẽ
          filled=True, # Tô màu cho đẹp và dễ biết kết quả
          feature_names=ten_dac_trung, # Ghi tên đặc trưng (Mỏ, Chân)
          class_names=ten_loai, # Ghi tên lớp (Vịt, Gà)
          rounded=True, # Bo tròn góc ô cho đẹp
          fontsize=12) # Chữ to hơn chút
plt.title(f"Cây Quyết Định Gà/Vịt (Độ sâu tối đa = {may_hoc.max_depth})") # Tiêu đề
plt.show() # Hiện hình cây lên

# === Bước 6: Kiểm tra xem máy học đoán giỏi cỡ nào ===
# (Lưu ý: Kiểm tra trên chính dữ liệu đã học thì thường điểm cao, không khách quan lắm)
print("--- Kiểm tra kết quả ---")
# Bảo máy đoán lại nhãn cho toàn bộ dữ liệu X
du_doan = may_hoc.predict(X)

# So sánh dự đoán với nhãn thật xem đúng bao nhiêu %
do_chinh_xac = accuracy_score(y, du_doan)
print(f"Tỷ lệ đoán đúng (trên dữ liệu đã học): {do_chinh_xac * 100:.2f}%") # In tỷ lệ %

# In thêm thông tin chi tiết (không cần hiểu sâu cái này cũng được)
print("\nBáo cáo chi tiết:")
print(classification_report(y, du_doan, target_names=ten_loai))
print("-" * 30)

# === Bước 7: Thử đoán cho con vật mới ===
# Giờ mình có 2 con vật mới, chỉ biết số đo, thử hỏi máy xem nó là Gà hay Vịt

# Con 1: Mỏ 4.0cm, Chân 2.0cm (Trông giống Vịt)
# Con 2: Mỏ 2.0cm, Chân 4.0cm (Trông giống Gà)
mau_moi = np.array([
    [4.0, 2.0],
    [2.0, 4.0]
])

print("--- Thử đoán cho con vật mới ---")
# Đưa số đo 2 con mới cho máy đoán
ket_qua_doan = may_hoc.predict(mau_moi)

# Xem máy đoán là gì
for i in range(len(mau_moi)):
    so_do = mau_moi[i] # Lấy số đo con thứ i
    nhan_doan_duoc = ket_qua_doan[i] # Lấy nhãn máy đoán (0 hoặc 1)
    ten_loai_doan_duoc = ten_loai[nhan_doan_duoc] # Đổi số 0/1 thành chữ 'Vịt'/'Gà'
    print(f"Con vật có số đo {so_do}: Máy đoán là '{ten_loai_doan_duoc}'")