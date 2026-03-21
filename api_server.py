from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine # Thêm thư viện này
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Hệ Thống Gợi Ý Bài Tập - Cloud Server Edition")

# ==========================================
# 1. CẤU HÌNH KẾT NỐI DATABASE CHO CLOUD SERVER (LINUX)
# ==========================================
def get_db_connection():
    # Sử dụng pymssql thay cho pyodbc để chạy mượt trên Linux server
    db_url = "mssql+pymssql://userPersonalizedSystem:123456789@118.69.126.49/Data_PersonalizedSystem"
    engine = create_engine(db_url)
    return engine.connect()

# (Các phần load_data_from_sql và API bên dưới BẠN GIỮ NGUYÊN HOÀN TOÀN NHÉ)

def load_data_from_sql():
    try:
        conn = get_db_connection()
        
        # Đọc dữ liệu bài tập
        query_exercises = """
            SELECT 
                Id AS ExerciseID, 
                TenBaiTap AS Title, 
                ISNULL(MaMon, '') + ' ' + TenBaiTap AS Tags,
                ISNULL(MaDoKho, 1) AS Difficulty
            FROM BAITAP
        """
        df_exercises = pd.read_sql(query_exercises, conn)
        
        # Đọc lịch sử làm bài (ĐÃ ĐỔI THÀNH ĐIỂM SỐ)
        query_history = """
            SELECT 
                MaSinhVien AS StudentID, 
                MaBaiTap AS ExerciseID, 
                DiemSo AS Score 
            FROM AI_LichSuLamBai
        """
        df_history = pd.read_sql(query_history, conn)
        
        conn.close()
        return df_exercises, df_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy xuất dữ liệu Remote SQL: {str(e)}")

# ==========================================
# 2. LUỒNG SINH VIÊN: AI GỢI Ý BÀI TẬP 
# ==========================================
class RecommendRequest(BaseModel):
    student_id: int
    top_k: int = 3

@app.post("/api/recommend", tags=["Sinh Viên"])
def get_recommendations(
    request: RecommendRequest, 
    x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")
):
    if x_user_role != "student":
        raise HTTPException(status_code=403, detail="Cấm truy cập! Chỉ sinh viên mới được nhận gợi ý.")

    df_exercises, df_history = load_data_from_sql()
    df_exercises['Tags'] = df_exercises['Tags'].fillna('')
    
    # AI ĐÁNH GIÁ: Nếu điểm >= 5.0 thì coi như đã vượt qua bài đó
    passed_exercises = df_history[(df_history['StudentID'] == request.student_id) & (df_history['Score'] >= 5.0)]['ExerciseID'].tolist()
    
    # Xác định Năng lực hiện tại
    if not passed_exercises:
        easy_exercises = df_exercises[df_exercises['Difficulty'] == 1].head(request.top_k).to_dict('records')
        return {"status": "new_student", "current_level": 1, "message": "Gợi ý cơ bản", "recommendations": easy_exercises}
    else:
        current_level = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises)]['Difficulty'].max()
        if pd.isna(current_level): current_level = 1

    # Huấn luyện thuật toán
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_exercises['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises)].index.tolist()
    sim_scores = sum([cosine_sim[i] for i in indices])
    
    related_indices = sim_scores.argsort()[::-1]
    recommendations = []
    
    for idx in related_indices:
        row = df_exercises.iloc[idx]
        ex_id = row['ExerciseID']
        ex_diff = row['Difficulty']
        
        if ex_id not in passed_exercises and ex_diff <= current_level + 1:
            recommendations.append(row.to_dict())
            
        if len(recommendations) >= request.top_k: 
            break

    return {
        "status": "success", 
        "current_level": int(current_level),
        "recommendations": recommendations
    }

# ==========================================
# 3. API NỘP BÀI TẬP (NHẬN ĐIỂM SỐ 0-10)
# ==========================================
class SubmitResultRequest(BaseModel):
    student_id: int
    exercise_id: int
    score: float  # Nhận điểm số thập phân (VD: 8.5)

@app.post("/api/submit-result", tags=["Sinh Viên"])
def submit_exercise_result(
    request: SubmitResultRequest,
    x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")
):
    if x_user_role != "student":
        raise HTTPException(status_code=403, detail="Cấm truy cập!")
    
    # Chặn không cho nhập điểm vớ vẩn
    if request.score < 0 or request.score > 10:
        raise HTTPException(status_code=400, detail="Điểm số phải nằm trong khoảng từ 0 đến 10")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Lưu điểm số thẳng vào Database
        insert_query = "INSERT INTO AI_LichSuLamBai (MaSinhVien, MaBaiTap, DiemSo) VALUES (?, ?, ?)"
        cursor.execute(insert_query, (request.student_id, request.exercise_id, request.score))
        conn.commit() 
        conn.close()
        return {"status": "success", "message": f"Đã lưu thành công {request.score} điểm cho bài tập {request.exercise_id}!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lưu Database: {str(e)}")

# ==========================================
# 4. LUỒNG GIẢNG VIÊN (Thống kê theo Điểm)
# ==========================================
@app.get("/api/teacher/student/{student_id}", tags=["Giảng Viên"])
def get_student_profile(
    student_id: int, 
    x_user_role: str = Header(None, description="Bắt buộc nhập 'teacher'")
):
    if x_user_role != "teacher":
        raise HTTPException(status_code=403, detail="Chỉ giảng viên mới xem được.")

    df_exercises, df_history = load_data_from_sql()
    student_data = df_history[df_history['StudentID'] == student_id]
    
    if student_data.empty:
        return {"status": "error", "message": "Sinh viên chưa làm bài."}

    # Phân loại: >= 5.0 là Giỏi/Khá, < 5.0 là Yếu
    passed_ex = student_data[student_data['Score'] >= 5.0]['ExerciseID'].tolist()
    failed_ex = student_data[student_data['Score'] < 5.0]['ExerciseID'].tolist()

    strong_tags = df_exercises[df_exercises['ExerciseID'].isin(passed_ex)]['Tags'].explode().unique().tolist()
    weak_tags = df_exercises[df_exercises['ExerciseID'].isin(failed_ex)]['Tags'].explode().unique().tolist()
    weak_tags = [tag for tag in weak_tags if tag not in strong_tags]

    return {
        "student_id": student_id,
        "total_attempts": len(student_data),
        "good_score_count": len(passed_ex),
        "low_score_count": len(failed_ex),
        "strong_skills": strong_tags,
        "weak_skills": weak_tags
    }
    # ==========================================
# ==========================================
# ==========================================
# 7. API ĐĂNG NHẬP (LẤY DỮ LIỆU TỪ SQL)
# ==========================================
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login", tags=["Hệ Thống"])
def login_user(request: LoginRequest):
    try:
        conn = get_db_connection()
        # Tìm tài khoản trong bảng TAIKHOAN (Bạn sẽ cần tạo bảng này trong SQL Server)
        query = "SELECT VaiTro, MaNguoiDung, HoTen FROM TAIKHOAN WHERE TenDangNhap = ? AND MatKhau = ?"
        df_user = pd.read_sql(query, conn, params=(request.username, request.password))
        conn.close()

        if df_user.empty:
            return {"status": "error", "message": "Sai tên đăng nhập hoặc mật khẩu!"}
        
        user_info = df_user.iloc[0]
        return {
            "status": "success",
            "role": user_info['VaiTro'], # 'student' hoặc 'teacher'
            "user_id": int(user_info['MaNguoiDung']),
            "full_name": user_info['HoTen']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Database: {str(e)}")
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
