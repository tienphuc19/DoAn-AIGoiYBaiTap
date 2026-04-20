from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import json
import re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Hệ Thống Gợi Ý Bài Tập AI")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# KẾT NỐI DATABASE
def get_db_connection():
    db_url = "mssql+pymssql://userPersonalizedSystem:123456789@118.69.126.49/Data_PersonalizedSystem"
    return create_engine(db_url).connect()

def load_data_from_sql():
    try:
        conn = get_db_connection()
        query_exercises = "SELECT Id AS ExerciseID, TenBaiTap AS Title, ISNULL(MaMon, '') AS SubjectCode, ISNULL(MaMon, '') + ' ' + ISNULL(TenBaiTap, '') AS Tags, ISNULL(MaDoKho, 1) AS Difficulty FROM BAITAP"
        df_exercises = pd.read_sql(query_exercises, conn)
        query_history = "SELECT MaSinhVien AS StudentID, MaBaiTap AS ExerciseID, DiemSo AS Score FROM AI_LichSuLamBai"
        df_history = pd.read_sql(query_history, conn)
        conn.close()
        return df_exercises, df_history
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Lỗi SQL: {str(e)}")

# API CHI TIẾT BÀI TẬP
@app.get("/api/exercise-details/{ex_id}", tags=["Sinh Viên"])
def get_exercise_details(ex_id: int):
    conn = get_db_connection()
    mota, yeucau = "Đang cập nhật.", "Hoàn thành bài tập."
    try:
        df_bt = pd.read_sql(f"SELECT * FROM BAITAP WHERE Id = {ex_id}", conn)
        if not df_bt.empty:
            if 'MoTa' in df_bt.columns and pd.notna(df_bt['MoTa'].iloc[0]): 
                mota = str(df_bt['MoTa'].iloc[0])
            if 'YeuCau' in df_bt.columns and pd.notna(df_bt['YeuCau'].iloc[0]): 
                yeucau = str(df_bt['YeuCau'].iloc[0])
    except Exception: 
        pass
    conn.close()
    return {"status": "success", "mota": mota, "yeucau": yeucau}

# API CHẤM ĐIỂM AI (GEMINI-PRO)
class MockGradeRequest(BaseModel):
    student_id: int
    exercise_id: int
    submitted_work: str = "" 

@app.post("/api/mock-grade-and-submit", tags=["Sinh Viên"])
def grade_and_submit_result(request: MockGradeRequest, x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")):
    if x_user_role != "student": 
        raise HTTPException(status_code=403, detail="Cấm truy cập!")
    try:
        conn = get_db_connection()
        df_bt = pd.read_sql(f"SELECT TenBaiTap FROM BAITAP WHERE Id = {request.exercise_id}", conn)
        title = df_bt.iloc[0]['TenBaiTap'] if not df_bt.empty else "Bài tập lập trình"
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        code_sinh_vien = request.submitted_work.strip()
        
        if gemini_api_key and len(code_sinh_vien) > 15:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('gemini-pro')
                prompt = f"""Bạn là giảng viên chấm thi lập trình. Chấm bài: {title}
                Code sinh viên: {code_sinh_vien}
                Trả về JSON: {{"score": float, "criteria_eval": "nhận xét", "strengths": "ưu điểm", "weaknesses": "lỗi sai"}}"""
                response = model.generate_content(prompt)
                raw_text = response.text.strip()
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                ai_result = json.loads(match.group(0)) if match else {"score": 5.0, "criteria_eval": "Lỗi định dạng AI"}
                final_grade = float(ai_result.get("score", 0.0))
                feedback = ai_result
            except Exception as e:
                final_grade = 5.0
                feedback = {"criteria_eval": f"LỖI TỪ GOOGLE: {str(e)}", "strengths": "API gặp sự cố", "weaknesses": "Liên hệ admin"}
        else:
            final_grade = 0.0
            feedback = {"criteria_eval": "Code quá ngắn.", "strengths": "Trống", "weaknesses": "Trống"}

        query_upsert = text("""
            IF EXISTS (SELECT 1 FROM AI_LichSuLamBai WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id)
                UPDATE AI_LichSuLamBai SET DiemSo = :grade WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id
            ELSE
                INSERT INTO AI_LichSuLamBai (MaSinhVien, MaBaiTap, DiemSo) VALUES (:sv_id, :ex_id, :grade)
        """)
        conn.execute(query_upsert, {"sv_id": request.student_id, "ex_id": request.exercise_id, "grade": final_grade})
        conn.commit() 
        conn.close()
        return {"status": "success", "score": final_grade, "passed": final_grade >= 5.0, "feedback": feedback}
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

# THUẬT TOÁN GỢI Ý CỐT LÕI (CHỈ DÙNG CBF & ĐIỂM NĂNG LỰC CHÍNH XÁC)
class RecommendRequest(BaseModel):
    student_id: int
    top_k: int = 6
    subject_code: str = ""

@app.post("/api/recommend", tags=["Sinh Viên"])
def get_recommendations_cbf(request: RecommendRequest, x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")):
    if x_user_role != "student": 
        raise HTTPException(status_code=403, detail="Cấm truy cập!")
    
    fixed_avg_score = 0.0
    academic_rank = "Chưa có điểm"
    course_score_supa = 0.0
    supa_key = os.getenv("SUPABASE_KEY")
    
    # 1. Lấy điểm từ Supabase (Nền tảng khởi đầu)
    if supa_key:
        headers = {"apikey": supa_key, "Authorization": f"Bearer {supa_key}"}
        try:
            # Lấy Học lực chung
            res_int = requests.get("https://bxpugrlaosbemlfttrnk.supabase.co/rest/v1/integrated_scores", headers=headers, params={"student_id": f"eq.{request.student_id}"}, timeout=5)
            if res_int.status_code == 200 and len(res_int.json()) > 0:
                fixed_avg_score = float(res_int.json()[0].get('integrated_score', 0.0))
                academic_rank = str(res_int.json()[0].get('classification', ''))
            
            # Lấy Năng lực môn
            map_subj = {"CTDLGT": "CTDL", "OOP": "OOP", "NMLT": "NMLT", "KTLT": "KTLT"}
            res_c = requests.get("https://bxpugrlaosbemlfttrnk.supabase.co/rest/v1/course_scores", headers=headers, params={"student_id": f"eq.{request.student_id}", "course_code": f"eq.{map_subj.get(request.subject_code, '')}"}, timeout=5)
            if res_c.status_code == 200 and len(res_c.json()) > 0:
                course_score_supa = float(res_c.json()[0].get('score', 0.0))
        except: 
            pass

    df_exercises, df_history = load_data_from_sql()
    df_exercises['Tags'] = df_exercises['Tags'].fillna('')
    
    # 2. Bộ lọc môn học
    if request.subject_code:
        if request.subject_code == 'OOP': pattern = 'OOP|LTHDT|Lập trình hướng đối tượng'
        elif request.subject_code == 'CTDLGT': pattern = 'CTDL|Cấu trúc dữ liệu'
        elif request.subject_code == 'NMLT': pattern = 'NMLT|Nhập môn lập trình'
        elif request.subject_code == 'KTLT': pattern = 'KTLT|Kỹ thuật lập trình'
        else: pattern = request.subject_code
        df_exercises = df_exercises[df_exercises['SubjectCode'].str.contains(pattern, case=False, na=False)].reset_index(drop=True)

    subject_exercise_ids = df_exercises['ExerciseID'].tolist()
    student_history = df_history[(df_history['StudentID'] == request.student_id) & (df_history['ExerciseID'].isin(subject_exercise_ids))]
    
    completed_df = student_history[student_history['Score'] >= 5.0]
    completed_ids = completed_df['ExerciseID'].tolist()

    # 3. Tính Năng Lực Hiện Tại của môn
    # Đã tuân thủ: Nếu chưa làm bài AI nào thì lấy nguyên điểm môn Supabase (Dù là 0.0 vẫn giữ nguyên)
    if len(completed_df) > 0:
        current_comp = float(completed_df['Score'].mean())
    else:
        current_comp = float(course_score_supa)

    # 4. Xác định Target Difficulty trực tiếp từ Điểm Năng Lực
    if current_comp >= 8.0: target_diff = 3.0       # Giỏi -> Ưu tiên bài Khó
    elif current_comp >= 6.0: target_diff = 2.0     # Khá/TB -> Ưu tiên bài Trung bình
    else: target_diff = 1.0                         # Yếu/Chưa có điểm -> Ưu tiên bài Dễ

    candidate_ex = df_exercises[~df_exercises['ExerciseID'].isin(completed_ids)]
    
    if candidate_ex.empty: 
        return {
            "status": "success", 
            "avg_score": fixed_avg_score, 
            "academic_rank": academic_rank,
            "subject_score": round(current_comp, 1), 
            "recommendations": []
        }

    # 5. Thuật toán lọc dựa trên nội dung (CBF)
    tfidf = TfidfVectorizer()
    tf_matrix = tfidf.fit_transform(df_exercises['Tags'])
    cos_sim = cosine_similarity(tf_matrix, tf_matrix)
    passed_idx = df_exercises[df_exercises['ExerciseID'].isin(completed_ids)].index.tolist()
    
    final_list = []
    if passed_idx:
        scores_cb = sum([cos_sim[i] for i in passed_idx])
        if type(scores_cb) != list and scores_cb.max() > scores_cb.min(): 
            scores_cb = (scores_cb - scores_cb.min()) / (scores_cb.max() - scores_cb.min())
        
        dict_cb = {df_exercises.iloc[idx]['ExerciseID']: scores_cb[idx] for idx in df_exercises.index}
        
        for _, r in candidate_ex.iterrows():
            penalty = abs(r['Difficulty'] - target_diff) * 0.4
            final_score = dict_cb.get(r['ExerciseID'], 0) - penalty
            final_list.append({"ex": r.to_dict(), "score": float(final_score)})
    else:
        # Xử lý Cold-Start
        for _, r in candidate_ex.iterrows():
            final_score = 1.0 / (abs(r['Difficulty'] - target_diff) + 1.0)
            final_list.append({"ex": r.to_dict(), "score": float(final_score)})

    sorted_res = sorted(final_list, key=lambda x: x['score'], reverse=True)
    
    return {
        "status": "success", 
        "avg_score": fixed_avg_score, 
        "academic_rank": academic_rank,
        "subject_score": round(current_comp, 1), 
        "recommendations": [i['ex'] for i in sorted_res[:request.top_k]]
    }

# API LỊCH SỬ
@app.get("/api/history/{student_id}", tags=["Sinh Viên"])
def get_student_history(student_id: int, x_user_role: str = Header(None)):
    if x_user_role != "student": 
        raise HTTPException(status_code=403)
    conn = get_db_connection()
    query = text("SELECT h.MaBaiTap AS ExerciseID, b.TenBaiTap AS Title, h.DiemSo AS Score, b.MaDoKho AS Difficulty FROM AI_LichSuLamBai h JOIN BAITAP b ON h.MaBaiTap = b.Id WHERE h.MaSinhVien = :sv_id ORDER BY h.DiemSo DESC")
    df = pd.read_sql(query, conn, params={"sv_id": student_id})
    conn.close()
    return {"status": "success", "history": df.to_dict(orient="records")}

# API ĐĂNG NHẬP
class LoginRequest(BaseModel): 
    username: str
    password: str

@app.post("/api/login", tags=["Hệ Thống"])
def login_user(request: LoginRequest):
    conn = get_db_connection()
    df = pd.read_sql(text("SELECT VaiTro, MaNguoiDung, HoTen FROM TAIKHOAN WHERE TenDangNhap = :u AND MatKhau = :p"), conn, params={"u": request.username, "p": request.password})
    conn.close()
    if df.empty: 
        return {"status": "error", "message": "Sai thông tin!"}
    return {
        "status": "success", 
        "role": df.iloc[0]['VaiTro'], 
        "user_id": int(df.iloc[0]['MaNguoiDung']), 
        "full_name": df.iloc[0]['HoTen'], 
        "username": request.username
    }

# API GIẢNG VIÊN
@app.get("/api/teacher/overview", tags=["Giảng Viên"])
def get_teacher_overview(x_user_role: str = Header(None)):
    if x_user_role != "teacher": 
        raise HTTPException(status_code=403)
    conn = get_db_connection()
    df_sv = pd.read_sql("SELECT MaNguoiDung, TenDangNhap, HoTen FROM TAIKHOAN WHERE VaiTro = 'student'", conn)
    df_diem = pd.read_sql("SELECT MaSinhVien, DiemSo FROM AI_LichSuLamBai", conn)
    conn.close()
    
    df_sv['Lop'] = df_sv['HoTen'].str.extract(r'\((.*?)\)')[0].fillna('Chưa rõ')
    failed = df_diem[df_diem['DiemSo'] < 5.0].groupby('MaSinhVien').size().reset_index(name='f')
    df_sv = pd.merge(df_sv, failed, left_on='MaNguoiDung', right_on='MaSinhVien', how='left').fillna(0)
    
    classes = []
    for name, gp in df_sv.groupby('Lop'):
        sts = gp.apply(lambda r: {"user_id": int(r['MaNguoiDung']), "username": r['TenDangNhap'], "fullname": r['HoTen'], "failed_count": int(r['f'])}, axis=1).tolist()
        classes.append({"class_name": name, "student_count": len(sts), "students": sts})
    
    return {"status": "success", "weak_students_count": int((df_sv['f'] > 0).sum()), "classes": classes}

@app.get("/")
@app.head("/")
def serve_frontend(): 
    return FileResponse("index.html") if os.path.exists("index.html") else {"m": "No index.html"}
