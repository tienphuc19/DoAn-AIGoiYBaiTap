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
import requests
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env (nếu chạy ở máy cá nhân)
load_dotenv()

app = FastAPI(title="Hệ Thống Gợi Ý Bài Tập - Hoàn Thiện")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# CẤU HÌNH DATABASE SQL SERVER
# ==========================================
def get_db_connection():
    db_url = "mssql+pymssql://userPersonalizedSystem:123456789@118.69.126.49/Data_PersonalizedSystem"
    engine = create_engine(db_url)
    return engine.connect()

def load_data_from_sql():
    try:
        conn = get_db_connection()
        query_exercises = """
            SELECT Id AS ExerciseID, TenBaiTap AS Title, ISNULL(MaMon, '') AS SubjectCode,
                   ISNULL(MaMon, '') + ' ' + ISNULL(TenBaiTap, '') AS Tags, ISNULL(MaDoKho, 1) AS Difficulty
            FROM BAITAP
        """
        df_exercises = pd.read_sql(query_exercises, conn)
        query_history = "SELECT MaSinhVien AS StudentID, MaBaiTap AS ExerciseID, DiemSo AS Score FROM AI_LichSuLamBai"
        df_history = pd.read_sql(query_history, conn)
        conn.close()
        return df_exercises, df_history
    except Exception as e: raise HTTPException(status_code=500, detail=f"Lỗi SQL: {str(e)}")

# ==========================================
# API LẤY CHI TIẾT BÀI TẬP VÀ CHẤM ĐIỂM
# ==========================================
@app.get("/api/exercise-details/{ex_id}", tags=["Sinh Viên"])
def get_exercise_details(ex_id: int):
    conn = get_db_connection()
    mota = "Đang cập nhật mô tả từ cơ sở dữ liệu."
    yeucau = "1. Hoàn thành bài tập theo yêu cầu đề bài."
    criteria_list = []

    try:
        df_bt = pd.read_sql(f"SELECT * FROM BAITAP WHERE Id = {ex_id}", conn)
        if not df_bt.empty:
            if 'MoTa' in df_bt.columns and pd.notna(df_bt['MoTa'].iloc[0]) and str(df_bt['MoTa'].iloc[0]).strip() != "": mota = str(df_bt['MoTa'].iloc[0])
            if 'YeuCau' in df_bt.columns and pd.notna(df_bt['YeuCau'].iloc[0]) and str(df_bt['YeuCau'].iloc[0]).strip() != "": yeucau = str(df_bt['YeuCau'].iloc[0])

        df_tc = pd.read_sql("SELECT * FROM TIEUCHI_DANGBAI", conn)
        df_tc_filtered = pd.DataFrame()
        if 'MaBaiTap' in df_tc.columns: df_tc_filtered = df_tc[df_tc['MaBaiTap'] == ex_id]
        elif 'MaDangBai' in df_tc.columns and 'MaDangBai' in df_bt.columns: df_tc_filtered = df_tc[df_tc['MaDangBai'] == df_bt['MaDangBai'].iloc[0]]

        if not df_tc_filtered.empty:
            name_col = next((col for col in ['TenTieuChi', 'NoiDung', 'TieuChi'] if col in df_tc.columns), None)
            score_col = next((col for col in ['Diem', 'TrongSo', 'Score'] if col in df_tc.columns), None)
            for _, row in df_tc_filtered.iterrows():
                name = str(row[name_col]) if name_col else "Tiêu chí thành phần"
                score = str(int(row[score_col])) if score_col and pd.notna(row[score_col]) else "0"
                criteria_list.append({"name": name, "score": score})
    except Exception as e: print(f"Lỗi SQL: {e}")

    if len(criteria_list) == 0:
        criteria_list = [{"name": "Giải thuật chính xác, đáp ứng yêu cầu", "score": 40}, {"name": "Tối ưu độ phức tạp", "score": 30}, {"name": "Tuân thủ Coding convention", "score": 30}]

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
        final_grade = round(random.uniform(3.0, 9.5), 1) 
        query_upsert = text("""
            IF EXISTS (SELECT 1 FROM AI_LichSuLamBai WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id)
                UPDATE AI_LichSuLamBai SET DiemSo = :grade WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id
            ELSE
                INSERT INTO AI_LichSuLamBai (MaSinhVien, MaBaiTap, DiemSo) VALUES (:sv_id, :ex_id, :grade)
        """)
        conn.execute(query_upsert, {"sv_id": request.student_id, "ex_id": request.exercise_id, "grade": final_grade})
        conn.commit() 
        conn.close()
        return {"status": "success", "score": final_grade, "passed": final_grade >= 5.0, "message": f"🤖 Đã chấm tự động."}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# API LÕI: GỢI Ý BÀI TẬP (CONTENT-BASED FILTERING CHUẨN MỰC)
# ==========================================
class RecommendRequest(BaseModel):
    student_id: int
    top_k: int = 6
    subject_code: str = ""

@app.post("/api/recommend", tags=["Sinh Viên"])
def get_recommendations_cbf(request: RecommendRequest, x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")):
    if x_user_role != "student": raise HTTPException(status_code=403, detail="Cấm truy cập!")
    
    fixed_avg_score = 0.0
    academic_rank = "Chưa có điểm"
    course_score = 0.0 
    
    # 1. GIAO TIẾP VỚI SUPABASE (Lấy điểm hiển thị và điểm nền)
    supa_key = os.getenv("SUPABASE_KEY")
    if supa_key:
        headers = {"apikey": supa_key, "Authorization": f"Bearer {supa_key}"}
        
        # Lấy Học Lực Tổng
        try:
            res_int = requests.get("https://bxpugrlaosbemlfttrnk.supabase.co/rest/v1/integrated_scores", headers=headers, params={"student_id": f"eq.{request.student_id}"}, timeout=5)
            if res_int.status_code == 200 and len(res_int.json()) > 0:
                fixed_avg_score = round(float(res_int.json()[0].get('integrated_score', 0.0)), 1)
                academic_rank = str(res_int.json()[0].get('classification', 'Chưa có điểm'))
        except Exception: pass

        # Lấy Điểm Môn Học (Chuyển mã môn web sang mã Supabase)
        map_subject = {"CTDLGT": "CTDL", "OOP": "OOP", "NMLT": "NMLT", "KTLT": "KTLT"}
        supa_subject_code = map_subject.get(request.subject_code, request.subject_code)
        if request.subject_code:
            try:
                res_course = requests.get("https://bxpugrlaosbemlfttrnk.supabase.co/rest/v1/course_scores", headers=headers, params={"student_id": f"eq.{request.student_id}", "course_code": f"eq.{supa_subject_code}"}, timeout=5)
                if res_course.status_code == 200 and len(res_course.json()) > 0:
                    course_score = float(res_course.json()[0].get('score', 0.0))
            except Exception: pass

    # 2. XỬ LÝ DỮ LIỆU TỪ SQL SERVER
    df_exercises, df_history = load_data_from_sql()
    df_exercises['Tags'] = df_exercises['Tags'].fillna('')
    if request.subject_code:
        df_exercises = df_exercises[df_exercises['SubjectCode'].str.contains(request.subject_code, case=False, na=False)].reset_index(drop=True)
    
    student_history = df_history[df_history['StudentID'] == request.student_id]
    
    if df_exercises.empty: 
        return {"status": "success", "cf_error_message": "Chưa có dữ liệu bài tập trong SQL.", "avg_score": fixed_avg_score, "academic_rank": academic_rank, "recommendations": []}

    all_attempts_ids = student_history['ExerciseID'].tolist()
    
    # Lấy các bài đã qua môn (điểm >= 5.0) và khớp mã bài với môn học hiện tại
    passed_exercises_ids = student_history[
        (student_history['Score'] >= 5.0) & 
        (student_history['ExerciseID'].isin(df_exercises['ExerciseID']))
    ]['ExerciseID'].tolist()
    # 3. LOGIC CÁ NHÂN HÓA TRÌNH ĐỘ (DYNAMIC LEVEL)
    str_student_id = str(request.student_id)
    
    if passed_exercises_ids:
        # Đã có lịch sử trong SQL -> Trình độ = Mức độ khó cao nhất đã vượt qua
        current_level = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises_ids)]['Difficulty'].max()
    else:
        # CHƯA CÓ LỊCH SỬ (Cold Start Problem)
        if str_student_id.startswith("125"):
            # Sinh viên Khóa 25 (Năm 1): Mặc định Tân binh Level 1
            current_level = 1
        else:
            # Khóa cũ: Dựa vào điểm môn học từ Supabase để phân luồng
            if course_score >= 8.0: current_level = 3
            elif course_score >= 6.5: current_level = 2
            else: current_level = 1

    if pd.isna(current_level): current_level = 1

    # Lọc bài tập: Chưa làm & Độ khó phù hợp
    candidate_exercises = df_exercises[(~df_exercises['ExerciseID'].isin(all_attempts_ids)) & (df_exercises['Difficulty'] <= current_level + 1)]
    if candidate_exercises.empty: 
        return {"status": "success", "cf_error_message": "Tuyệt vời! Bạn đã hoàn thành lộ trình môn này.", "avg_score": fixed_avg_score, "academic_rank": academic_rank, "recommendations": []}

    # 4. THUẬT TOÁN CONTENT-BASED FILTERING
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_exercises['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    passed_indices = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises_ids)].index.tolist()
    
   final_recommendations_with_score = []
    if passed_indices:
        # TRƯỜNG HỢP 1: ĐÃ CÓ LỊCH SỬ LÀM BÀI
        sim_scores_cb = sum([cosine_sim[i] for i in passed_indices])
        if type(sim_scores_cb) != list and sim_scores_cb.max() > sim_scores_cb.min():
            sim_scores_cb = (sim_scores_cb - sim_scores_cb.min()) / (sim_scores_cb.max() - sim_scores_cb.min())
            
        cb_scores_dict = {df_exercises.iloc[idx]['ExerciseID']: sim_scores_cb[idx] for idx in df_exercises.index}
        
        for _, row in candidate_exercises.iterrows():
            sim_score = cb_scores_dict.get(row['ExerciseID'], 0)
            
            # 1. Phạt điểm nếu bài quá lệch so với Level (Ép AI chọn bài đúng trình độ)
            diff_penalty = abs(row['Difficulty'] - current_level) * 0.3 
            
            # 2. Cộng hệ số Random (0.01 -> 0.15) để phá Bong bóng lọc, tăng đa dạng
            diversity_bonus = random.uniform(0.01, 0.15) 
            
            final_score = sim_score - diff_penalty + diversity_bonus
            final_recommendations_with_score.append({"exercise": row.to_dict(), "final_score": float(final_score)})
    else:
        # TRƯỜNG HỢP 2: CHƯA CÓ LỊCH SỬ (GIẢI QUYẾT LỖI TOÀN BÀI DỄ)
        for _, row in candidate_exercises.iterrows():
            # Chấm điểm cao nhất cho bài tập có Độ khó KHỚP với Level hiện tại
            diff_match_score = 1.0 / (abs(row['Difficulty'] - current_level) + 1.0)
            
            # Trộn ngẫu nhiên để các dạng bài khác nhau được ngoi lên
            diversity_bonus = random.uniform(0.01, 0.2) 
            
            final_score = diff_match_score + diversity_bonus
            final_recommendations_with_score.append({"exercise": row.to_dict(), "final_score": float(final_score)})

    # Sắp xếp từ điểm cao xuống thấp và lấy Top K
    sorted_recommendations = sorted(final_recommendations_with_score, key=lambda x: x['final_score'], reverse=True)
    
    return {
        "status": "success", 
        "current_level": int(current_level), 
        "avg_score": float(fixed_avg_score), 
        "academic_rank": academic_rank,
        "subject_score": round(float(course_score), 1), 
        "recommendations": [item['exercise'] for item in sorted_recommendations[:request.top_k]]
    }
    }

# ==========================================
# CÁC API KHÁC (Lịch sử, Login, Dashboard)
# ==========================================
@app.get("/api/history/{student_id}", tags=["Sinh Viên"])
def get_student_history(student_id: int, x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")):
    if x_user_role != "student": raise HTTPException(status_code=403, detail="Cấm truy cập!")
    try:
        conn = get_db_connection()
        query = text("""
            SELECT h.MaBaiTap AS ExerciseID, b.TenBaiTap AS Title, ISNULL(b.MaMon, 'Không rõ') AS SubjectCode,
                   h.DiemSo AS Score, ISNULL(b.MaDoKho, 1) AS Difficulty
            FROM AI_LichSuLamBai h JOIN BAITAP b ON h.MaBaiTap = b.Id
            WHERE h.MaSinhVien = :sv_id ORDER BY h.DiemSo DESC
        """)
        df_history = pd.read_sql(query, conn, params={"sv_id": student_id})
        conn.close()
        history_list = [{"ExerciseID": int(r['ExerciseID']), "Title": str(r['Title']), "SubjectCode": str(r['SubjectCode']),
                         "Score": float(r['Score']), "Difficulty": int(r['Difficulty'])} for _, r in df_history.iterrows()]
        return {"status": "success", "history": history_list}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

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
        df_sv = pd.read_sql("SELECT MaNguoiDung, TenDangNhap, HoTen FROM TAIKHOAN WHERE VaiTro = 'student'", conn)
        df_diem = pd.read_sql("SELECT MaSinhVien, DiemSo FROM AI_LichSuLamBai", conn)
        conn.close()

        if df_sv.empty: return {"status": "success", "weak_students_count": 0, "classes": []}
        df_sv['Lop'] = df_sv['HoTen'].str.extract(r'\((.*?)\)')[0].fillna('Không xác định')

        if not df_diem.empty:
            failed_counts = df_diem[df_diem['DiemSo'] < 5.0].groupby('MaSinhVien').size().reset_index(name='FailedCount')
            df_sv = pd.merge(df_sv, failed_counts, left_on='MaNguoiDung', right_on='MaSinhVien', how='left')
        else:
            df_sv['FailedCount'] = 0

        df_sv['FailedCount'] = df_sv['FailedCount'].fillna(0).astype(int)
        weak_count = int((df_sv['FailedCount'] > 0).sum())

        classes_data = []
        for class_name, group in df_sv.groupby('Lop'):
            students = []
            for _, row in group.iterrows():
                students.append({"user_id": int(row['MaNguoiDung']), "username": str(row['TenDangNhap']), "fullname": str(row['HoTen']), "failed_count": int(row['FailedCount'])})
            classes_data.append({"class_name": str(class_name), "student_count": len(students), "students": students})

        return {"status": "success", "weak_students_count": weak_count, "classes": classes_data}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def serve_frontend():
    return FileResponse("index.html") if os.path.exists("index.html") else {"message": "Upload index.html!"}
