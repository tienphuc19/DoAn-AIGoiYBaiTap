from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI(title="Hệ Thống Gợi Ý Bài Tập")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    db_url = "mssql+pymssql://userPersonalizedSystem:123456789@118.69.126.49/Data_PersonalizedSystem"
    engine = create_engine(db_url)
    return engine.connect()

def load_data_from_sql():
    try:
        conn = get_db_connection()
        # Đã thêm MaMon vào truy vấn để AI biết bài đó thuộc môn nào
        query_exercises = """
            SELECT 
                Id AS ExerciseID, 
                TenBaiTap AS Title, 
                ISNULL(MaMon, '') AS SubjectCode,
                ISNULL(MaMon, '') + ' ' + TenBaiTap AS Tags,
                ISNULL(MaDoKho, 1) AS Difficulty
            FROM BAITAP
        """
        df_exercises = pd.read_sql(query_exercises, conn)
        
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

class RecommendRequest(BaseModel):
    student_id: int
    top_k: int = 6
    subject_code: str = ""  # Nhận mã môn học từ giao diện gửi lên

@app.post("/api/recommend", tags=["Sinh Viên"])
def get_recommendations(
    request: RecommendRequest, 
    x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")
):
    if x_user_role != "student":
        raise HTTPException(status_code=403, detail="Cấm truy cập!")

    df_exercises, df_history = load_data_from_sql()
    df_exercises['Tags'] = df_exercises['Tags'].fillna('')
    
    # LỌC NGAY TỪ ĐẦU: Chỉ lấy các bài tập thuộc đúng môn học được chọn
    if request.subject_code:
        df_exercises = df_exercises[df_exercises['SubjectCode'].str.contains(request.subject_code, case=False, na=False)]
    
    if df_exercises.empty:
        return {"status": "success", "current_level": 1, "recommendations": []}

    passed_exercises = df_history[(df_history['StudentID'] == request.student_id) & (df_history['Score'] >= 5.0)]['ExerciseID'].tolist()
    
    if not passed_exercises:
        easy_exercises = df_exercises[df_exercises['Difficulty'] <= 2].head(request.top_k).to_dict('records')
        return {"status": "new_student", "current_level": 1, "message": "Gợi ý cơ bản", "recommendations": easy_exercises}
    else:
        current_level = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises)]['Difficulty'].max()
        if pd.isna(current_level): current_level = 1

    # Chạy Content-Based Filtering trên đúng danh sách bài tập của môn đó
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
        
        if ex_id not in passed_exercises and ex_diff <= current_level + 2:
            recommendations.append(row.to_dict())
            
        if len(recommendations) >= request.top_k: 
            break

    return {
        "status": "success", 
        "current_level": int(current_level),
        "recommendations": recommendations
    }

class SubmitResultRequest(BaseModel):
    student_id: int
    exercise_id: int
    score: float

@app.post("/api/submit-result", tags=["Sinh Viên"])
def submit_exercise_result(
    request: SubmitResultRequest,
    x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")
):
    if x_user_role != "student":
        raise HTTPException(status_code=403, detail="Cấm truy cập!")
    try:
        conn = get_db_connection()
        insert_query = text("INSERT INTO AI_LichSuLamBai (MaSinhVien, MaBaiTap, DiemSo) VALUES (:sv, :bt, :diem)")
        conn.execute(insert_query, {"sv": request.student_id, "bt": request.exercise_id, "diem": request.score})
        conn.commit() 
        conn.close()
        return {"status": "success", "message": "Đã lưu thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lưu Database: {str(e)}")

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login", tags=["Hệ Thống"])
def login_user(request: LoginRequest):
    try:
        conn = get_db_connection()
        query = text("SELECT VaiTro, MaNguoiDung, HoTen FROM TAIKHOAN WHERE TenDangNhap = :usr AND MatKhau = :pwd")
        df_user = pd.read_sql(query, conn, params={"usr": request.username, "pwd": request.password})
        conn.close()

        if df_user.empty:
            return {"status": "error", "message": "Sai tên đăng nhập hoặc mật khẩu!"}
        
        user_info = df_user.iloc[0]
        return {
            "status": "success",
            "role": user_info['VaiTro'],
            "user_id": int(user_info['MaNguoiDung']),
            "full_name": user_info['HoTen']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Database: {str(e)}")

@app.get("/", tags=["Hệ Thống"])
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Vui lòng upload index.html lên GitHub!"}
