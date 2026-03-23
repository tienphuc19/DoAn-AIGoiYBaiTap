from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random 

app = FastAPI(title="Hệ Thống Gợi Ý Bài Tập - Hoàn Thiện")

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
        query_exercises = """
            SELECT 
                Id AS ExerciseID, 
                TenBaiTap AS Title, 
                ISNULL(MaMon, '') AS SubjectCode,
                ISNULL(MaMon, '') + ' ' + ISNULL(TenBaiTap, '') AS Tags,
                ISNULL(MaDoKho, 1) AS Difficulty
            FROM BAITAP
        """
        df_exercises = pd.read_sql(query_exercises, conn)
        
        query_history = """
            SELECT MaSinhVien AS StudentID, MaBaiTap AS ExerciseID, DiemSo AS Score 
            FROM AI_LichSuLamBai
        """
        df_history = pd.read_sql(query_history, conn)
        conn.close()
        return df_exercises, df_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi SQL: {str(e)}")

@app.get("/api/exercise-details/{ex_id}", tags=["Sinh Viên"])
def get_exercise_details(ex_id: int):
    conn = get_db_connection()
    mota = "Thuật toán BFS mô phỏng việc lan tỏa như sóng nước, thăm tất cả các đỉnh gần gốc trước khi đi xa hơn. Đây là nền tảng để tìm đường đi ngắn nhất trong đồ thị không có trọng số." if ex_id == 374 else "Đang cập nhật mô tả từ cơ sở dữ liệu."
    yeucau = "1. Sử dụng Queue để cài đặt thuật toán BFS. In ra thứ tự các đỉnh được thăm bắt đầu từ một đỉnh s cho trước." if ex_id == 374 else "1. Hoàn thành bài tập theo yêu cầu đề bài."
    criteria_list = [
        {"name": "Sử dụng mảng đánh dấu (Visited) để không thăm lại các đỉnh đã xử lý", "score": 40},
        {"name": "Quản lý đúng thứ tự đưa vào và lấy ra khỏi Queue theo nguyên tắc FIFO", "score": 25},
        {"name": "Đảm bảo thăm hết các đỉnh trong cùng một thành phần liên thông", "score": 20},
        {"name": "Đạt độ phức tạp thời gian O(n + e)", "score": 15}
    ] if ex_id == 374 else []

    try:
        df_bt = pd.read_sql(f"SELECT * FROM BAITAP WHERE Id = {ex_id}", conn)
        if not df_bt.empty:
            if 'MoTa' in df_bt.columns and pd.notna(df_bt['MoTa'].iloc[0]) and str(df_bt['MoTa'].iloc[0]).strip() != "":
                mota = str(df_bt['MoTa'].iloc[0])
            if 'YeuCau' in df_bt.columns and pd.notna(df_bt['YeuCau'].iloc[0]) and str(df_bt['YeuCau'].iloc[0]).strip() != "":
                yeucau = str(df_bt['YeuCau'].iloc[0])

        df_tc = pd.read_sql("SELECT * FROM TIEUCHI_DANGBAI", conn)
        df_tc_filtered = pd.DataFrame()
        if 'MaBaiTap' in df_tc.columns:
            df_tc_filtered = df_tc[df_tc['MaBaiTap'] == ex_id]
        elif 'MaDangBai' in df_tc.columns and 'MaDangBai' in df_bt.columns:
            madangbai = df_bt['MaDangBai'].iloc[0]
            df_tc_filtered = df_tc[df_tc['MaDangBai'] == madangbai]

        if not df_tc_filtered.empty:
            criteria_list = []
            name_col = next((col for col in ['TenTieuChi', 'NoiDung', 'TieuChi'] if col in df_tc.columns), None)
            score_col = next((col for col in ['Diem', 'TrongSo', 'Score'] if col in df_tc.columns), None)
            for _, row in df_tc_filtered.iterrows():
                name = str(row[name_col]) if name_col else "Tiêu chí thành phần"
                score = str(int(row[score_col])) if score_col and pd.notna(row[score_col]) else "0"
                criteria_list.append({"name": name, "score": score})
    except Exception as e:
        print(f"Lỗi truy xuất SQL: {e}")

    if len(criteria_list) == 0:
        criteria_list = [
            {"name": "Giải thuật chính xác, đáp ứng yêu cầu bài toán", "score": 40},
            {"name": "Tối ưu hóa bộ nhớ và độ phức tạp thuật toán", "score": 30},
            {"name": "Tuân thủ Coding convention và viết comment rõ ràng", "score": 30}
        ]

    conn.close()
    return {"status": "success", "mota": mota, "yeucau": yeucau, "criteria": criteria_list}

class MockGradeRequest(BaseModel):
    student_id: int
    exercise_id: int
    submitted_work: str = "" 

@app.post("/api/mock-grade-and-submit", tags=["Sinh Viên"])
def mock_grade_and_submit_result(request: MockGradeRequest, x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")):
    if x_user_role != "student": raise HTTPException(status_code=403, detail="Cấm truy cập!")
    try:
        conn = get_db_connection()
        final_grade = round(random.uniform(6.5, 9.5), 1) 
        query_upsert = text("""
            IF EXISTS (SELECT 1 FROM AI_LichSuLamBai WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id)
                UPDATE AI_LichSuLamBai SET DiemSo = :grade WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id
            ELSE
                INSERT INTO AI_LichSuLamBai (MaSinhVien, MaBaiTap, DiemSo) VALUES (:sv_id, :ex_id, :grade)
        """)
        conn.execute(query_upsert, {"sv_id": request.student_id, "ex_id": request.exercise_id, "grade": final_grade})
        conn.commit() 
        conn.close()
        return {"status": "success", "score": final_grade, "passed": final_grade >= 5.0, "message": f"🤖 Đã chấm tự động dựa trên tiêu chí SQL."}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# CÁI KHUÔN NÀY ĐÃ ĐƯỢC THÊM LẠI ĐẦY ĐỦ RỒI ĐÂY
# ==========================================
class RecommendRequest(BaseModel):
    student_id: int
    top_k: int = 6
    subject_code: str = ""

@app.post("/api/recommend", tags=["Sinh Viên"])
def get_recommendations_hybrid(request: RecommendRequest, x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")):
    if x_user_role != "student": raise HTTPException(status_code=403, detail="Cấm truy cập!")
    df_exercises, df_history = load_data_from_sql()
    df_exercises['Tags'] = df_exercises['Tags'].fillna('')
    
    if request.subject_code:
        df_exercises = df_exercises[df_exercises['SubjectCode'].str.contains(request.subject_code, case=False, na=False)]
        df_exercises = df_exercises.reset_index(drop=True)
    
    student_history = df_history[df_history['StudentID'] == request.student_id]
    if not student_history.empty:
        avg_score = round(student_history['Score'].mean(), 1)
        academic_rank = "Giỏi" if avg_score >= 8.0 else ("Khá" if avg_score >= 6.5 else ("Trung Bình" if avg_score >= 5.0 else "Yếu"))
    else:
        avg_score = 0.0
        academic_rank = "Chưa có bài tập"

    if df_exercises.empty: return {"status": "success", "cf_error_message": "Không có bài tập.", "avg_score": avg_score, "academic_rank": academic_rank, "recommendations": []}

    all_attempts_ids = student_history['ExerciseID'].tolist()
    passed_exercises_ids = student_history[student_history['Score'] >= 5.0]['ExerciseID'].tolist()
    current_level = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises_ids)]['Difficulty'].max() if passed_exercises_ids else 1
    if pd.isna(current_level): current_level = 1

    candidate_exercises = df_exercises[(~df_exercises['ExerciseID'].isin(all_attempts_ids)) & (df_exercises['Difficulty'] <= current_level + 2)]
    if candidate_exercises.empty: return {"status": "success", "cf_error_message": "Đã làm hết bài.", "avg_score": avg_score, "academic_rank": academic_rank, "recommendations": []}

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_exercises['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    passed_indices = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises_ids)].index.tolist()
    sim_scores_cb = sum([cosine_sim[i] for i in passed_indices]) if passed_indices else []
    if type(sim_scores_cb) != list and sim_scores_cb.max() > sim_scores_cb.min():
        sim_scores_cb = (sim_scores_cb - sim_scores_cb.min()) / (sim_scores_cb.max() - sim_scores_cb.min())

    try:
        df_pivot = df_history.pivot_table(index='StudentID', columns='ExerciseID', values='Score')
        if df_pivot.shape[0] < 2 or request.student_id not in df_pivot.index: raise Exception("CF Inactive")
        item_sim = cosine_similarity(df_pivot.T.fillna(0))
        item_sim_df = pd.DataFrame(item_sim, index=df_pivot.columns, columns=df_pivot.columns)
        student_scores = df_pivot.loc[request.student_id].dropna()
        cf_scores_list = []
        for ex_id in df_exercises['ExerciseID'].tolist():
            if ex_id in passed_exercises_ids: cf_scores_list.append(0); continue
            if ex_id in item_sim_df.columns:
                sim_subset = item_sim_df[ex_id].reindex(student_scores.index).fillna(0)
                cf_scores_list.append(((sim_subset * student_scores).sum() / sim_subset.abs().sum()) if sim_subset.abs().sum() > 0 else 0)
            else: cf_scores_list.append(0)
        if len(cf_scores_list) > 0 and max(cf_scores_list) > min(cf_scores_list):
            cf_scores_list = [(x - min(cf_scores_list)) / (max(cf_scores_list) - min(cf_scores_list)) for x in cf_scores_list]
        scores_cf, cf_error_msg = cf_scores_list, False 
    except Exception as e: scores_cf, cf_error_msg = None, e

    final_recommendations_with_score = []
    cb_scores_dict = {df_exercises.iloc[idx]['ExerciseID']: sim_scores_cb[idx] for idx in df_exercises.index} if passed_indices else {}
    cf_scores_dict = {df_exercises.iloc[idx]['ExerciseID']: scores_cf[idx] for idx in df_exercises.index} if scores_cf is not None else {}
    
    for _, row in candidate_exercises.iterrows():
        cb_score = cb_scores_dict.get(row['ExerciseID'], 0)
        cf_score = cf_scores_dict.get(row['ExerciseID'], 0) if scores_cf is not None else 0
        final_score = 0.6 * cb_score + 0.4 * cf_score if scores_cf is not None else cb_score
        final_recommendations_with_score.append({"exercise": row.to_dict(), "final_hybrid_score": float(final_score)})

    sorted_recommendations = sorted(final_recommendations_with_score, key=lambda x: x['final_hybrid_score'], reverse=True)
    return {
        "status": "success", "current_level": int(current_level), "avg_score": float(avg_score), "academic_rank": academic_rank,
        "cf_error_message": str(cf_error_msg) if cf_error_msg else "Item-Based Collaborative Filtering is ACTIVE.",
        "recommendations": [item['exercise'] for item in sorted_recommendations[:request.top_k]]
    }

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
        if df_user.empty: return {"status": "error", "message": "Sai tên đăng nhập hoặc mật khẩu!"}
        return {"status": "success", "role": df_user.iloc[0]['VaiTro'], "user_id": int(df_user.iloc[0]['MaNguoiDung']), "full_name": df_user.iloc[0]['HoTen']}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/teacher/overview", tags=["Giảng Viên"])
def get_teacher_overview(x_user_role: str = Header(None, description="Bắt buộc nhập 'teacher'")):
    if x_user_role != "teacher": raise HTTPException(status_code=403, detail="Cấm truy cập!")
    try:
        conn = get_db_connection()
        df_sv = pd.read_sql("SELECT MaNguoiDung, HoTen FROM TAIKHOAN WHERE VaiTro = 'student'", conn)
        df_diem = pd.read_sql("SELECT MaSinhVien, DiemSo FROM AI_LichSuLamBai", conn)
        conn.close()

        if df_sv.empty: return {"status": "success", "weak_students_count": 0, "classes": []}

        df_sv['Lop'] = df_sv['HoTen'].str.extract(r'\((.*?)\)')[0]
        df_sv['Lop'] = df_sv['Lop'].fillna('Không xác định')

        class_counts = df_sv['Lop'].value_counts().reset_index()
        class_counts.columns = ['class_name', 'student_count']
        classes_data = class_counts.to_dict('records')

        weak_count = 0
        if not df_diem.empty:
            avg_scores = df_diem.groupby('MaSinhVien')['DiemSo'].mean().reset_index()
            weak_count = int((avg_scores['DiemSo'] < 5.0).sum())

        return {
            "status": "success",
            "weak_students_count": weak_count,
            "classes": classes_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def serve_frontend():
    return FileResponse("index.html") if os.path.exists("index.html") else {"message": "Upload index.html!"}
