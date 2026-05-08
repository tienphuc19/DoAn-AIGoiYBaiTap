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
import random
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Hệ Thống Gợi Ý Bài Tập AI")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

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

@app.get("/api/test-diem/{student_id}", tags=["Debug"])
def test_diem_supabase(student_id: int):
    supa_url = os.getenv("SUPABASE_URL", "https://bxpugrlaosbemlfttrnk.supabase.co")
    supa_key = os.getenv("SUPABASE_KEY")
    if not supa_key: return {"error": "Chưa cài đặt SUPABASE_KEY trên Render"}

    headers = {"apikey": supa_key, "Authorization": f"Bearer {supa_key}"}
    
    res1 = requests.get(f"{supa_url}/rest/v1/integrated_scores", headers=headers, params={"student_id": f"eq.{student_id}"})
    if res1.status_code != 200 or len(res1.json()) == 0:
        res1 = requests.get(f"{supa_url}/rest/v1/integrated_scores", headers=headers, params={"student": f"eq.{student_id}"})
    
    res2 = requests.get(f"{supa_url}/rest/v1/course_scores", headers=headers, params={"student_id": f"eq.{student_id}"})

    return {
        "1. Mã Sinh Viên Đang Test": student_id,
        "2. Dữ liệu bảng Học Lực (integrated_scores)": res1.json() if res1.status_code == 200 else str(res1.text),
        "3. Dữ liệu bảng Điểm Môn (course_scores)": res2.json() if res2.status_code == 200 else str(res2.text),
        "Kết luận": "Nếu phần 3 hiện mảng rỗng [], nghĩa là nhóm kia CHƯA NHẬP điểm môn cho sinh viên này!"
    }

@app.get("/api/exercise-details/{ex_id}", tags=["Sinh Viên"])
def get_exercise_details(ex_id: int):
    conn = get_db_connection()
    mota, yeucau = "Đang cập nhật.", "Hoàn thành bài tập."
    try:
        df_bt = pd.read_sql(f"SELECT * FROM BAITAP WHERE Id = {ex_id}", conn)
        if not df_bt.empty:
            if 'MoTa' in df_bt.columns and pd.notna(df_bt['MoTa'].iloc[0]): mota = str(df_bt['MoTa'].iloc[0])
            if 'YeuCau' in df_bt.columns and pd.notna(df_bt['YeuCau'].iloc[0]): yeucau = str(df_bt['YeuCau'].iloc[0])
    except Exception: pass
    conn.close()
    return {"status": "success", "mota": mota, "yeucau": yeucau}

class MockGradeRequest(BaseModel):
    student_id: int; exercise_id: int; submitted_work: str = "" 

# ==============================================================================
# AI GEMINI 2.0 FLASH CHẤM ĐIỂM
# ==============================================================================
@app.post("/api/mock-grade-and-submit", tags=["Sinh Viên"])
def grade_and_submit_result(request: MockGradeRequest, x_user_role: str = Header(None)):
    if x_user_role != "student": raise HTTPException(status_code=403)
    
    if not request.submitted_work or len(request.submitted_work.strip()) < 5:
        return {"status": "error", "message": "Code quá ngắn hoặc trống, vui lòng viết code đàng hoàng!"}

    conn = get_db_connection()
    try:
        df_bt = pd.read_sql(f"SELECT TenBaiTap, MoTa, YeuCau FROM BAITAP WHERE Id = {request.exercise_id}", conn)
        if df_bt.empty: raise Exception("Không tìm thấy bài tập!")
            
        ten_bt = str(df_bt.iloc[0]['TenBaiTap'])
        mota = str(df_bt.iloc[0]['MoTa'])
        yeucau = str(df_bt.iloc[0]['YeuCau'])

        gemini_key = "AIzaSyDhRK9jFo69jzzB_LuVFA26Bg7HvzUb6ic"
        
        prompt = f"""
        Bạn là giảng viên chấm thi Lập trình nghiêm khắc. Chấm điểm code của sinh viên:
        - Tên: {ten_bt}
        - Yêu cầu: {yeucau}
        
        CODE SINH VIÊN NỘP:
        ```
        {request.submitted_work}
        ```
        Nhiệm vụ: Chấm trên thang điểm 10. Tìm lỗi sai hoặc khen ngợi.
        BẠN CHỈ ĐƯỢC PHÉP TRẢ VỀ DUY NHẤT MỘT CHUỖI JSON ĐÚNG ĐỊNH DẠNG DƯỚI ĐÂY, KHÔNG GIẢI THÍCH THÊM:
        {{
            "score": 8.5,
            "strengths": "[1 câu ngắn khen ngợi]",
            "weaknesses": "[1 câu ngắn chỉ lỗi sai]"
        }}
        """

        # ĐÃ SỬA: Thay tên thành gemini-2.0-flash lấy từ danh sách của bạn
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        ai_res = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload)
        
        if ai_res.status_code != 200:
            raise Exception(f"Lỗi từ hệ thống Google AI: {ai_res.text}")

        res_data = ai_res.json()
        ai_response_text = res_data['candidates'][0]['content']['parts'][0]['text']
        
        ai_response_text = ai_response_text.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(ai_response_text)
        
        final_grade = float(result_json.get("score", 0.0))
        feedback = {
            "strengths": result_json.get("strengths", "Code có cấu trúc cơ bản."),
            "weaknesses": result_json.get("weaknesses", "Chưa thể phân tích sâu hơn.")
        }

        query_upsert = text("""
            IF EXISTS (SELECT 1 FROM AI_LichSuLamBai WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id)
                UPDATE AI_LichSuLamBai SET DiemSo = :grade WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id
            ELSE INSERT INTO AI_LichSuLamBai (MaSinhVien, MaBaiTap, DiemSo) VALUES (:sv_id, :ex_id, :grade)
        """)
        conn.execute(query_upsert, {"sv_id": request.student_id, "ex_id": request.exercise_id, "grade": final_grade})
        conn.commit()
        
        return {"status": "success", "score": final_grade, "passed": final_grade >= 5.0, "feedback": feedback}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI trả về kết quả bị lỗi chữ, vui lòng nộp lại!")
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
# ==============================================================================

class RecommendRequest(BaseModel):
    student_id: int; top_k: int = 6; subject_code: str = ""

@app.post("/api/recommend", tags=["Sinh Viên"])
def get_recommendations_cbf(request: RecommendRequest, x_user_role: str = Header(None)):
    if x_user_role != "student": raise HTTPException(status_code=403)
    
    fixed_avg_score = 0.0; academic_rank = "Chưa có điểm"; course_score_supa = 0.0
    
    supa_url = os.getenv("SUPABASE_URL", "https://bxpugrlaosbemlfttrnk.supabase.co")
    supa_key = os.getenv("SUPABASE_KEY")
    
    if supa_key:
        headers = {"apikey": supa_key, "Authorization": f"Bearer {supa_key}"}
        
        try:
            res_int = requests.get(f"{supa_url}/rest/v1/integrated_scores", headers=headers, params={"student_id": f"eq.{request.student_id}"}, timeout=5)
            if res_int.status_code != 200 or len(res_int.json()) == 0:
                res_int = requests.get(f"{supa_url}/rest/v1/integrated_scores", headers=headers, params={"student": f"eq.{request.student_id}"}, timeout=5)
                
            if res_int.status_code == 200 and len(res_int.json()) > 0:
                data = res_int.json()[0]
                fixed_avg_score = float(data.get('integrated_score', 0.0))
                academic_rank = str(data.get('classification', ''))
        except Exception: pass

        try:
            map_subj = {"CTDLGT": "CTDL", "OOP": "OOP", "NMLT": "NMLT", "KTLT": "KTLT"}
            target_code = map_subj.get(request.subject_code, request.subject_code)
            
            res_c = requests.get(f"{supa_url}/rest/v1/course_scores", headers=headers, params={"student_id": f"eq.{request.student_id}"}, timeout=5)
            
            if res_c.status_code == 200 and len(res_c.json()) > 0:
                for item in res_c.json():
                    db_code = str(item.get('course_code', '')).strip().upper()
                    if db_code == target_code.upper():
                        course_score_supa = float(item.get('score', 0.0))
                        break
        except Exception: pass

    df_exercises, df_history = load_data_from_sql()
    df_exercises['Tags'] = df_exercises['Tags'].fillna('')
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
    
    if len(completed_df) > 0:
        current_comp = float(completed_df['Score'].mean())
    else:
        current_comp = float(course_score_supa)

    target_diff = 3.0 if current_comp >= 8.0 else (2.0 if current_comp >= 6.0 else 1.0)
    candidate_ex = df_exercises[~df_exercises['ExerciseID'].isin(completed_ids)]
    
    if candidate_ex.empty: 
        return {"status": "success", "avg_score": fixed_avg_score, "academic_rank": academic_rank, "subject_score": round(current_comp, 1), "recommendations": []}

    tfidf = TfidfVectorizer(); tf_matrix = tfidf.fit_transform(df_exercises['Tags']); cos_sim = cosine_similarity(tf_matrix, tf_matrix)
    passed_idx = df_exercises[df_exercises['ExerciseID'].isin(completed_ids)].index.tolist()
    
    final_list = []
    if passed_idx:
        scores_cb = sum([cos_sim[i] for i in passed_idx])
        if type(scores_cb) != list and scores_cb.max() > scores_cb.min(): scores_cb = (scores_cb - scores_cb.min()) / (scores_cb.max() - scores_cb.min())
        dict_cb = {df_exercises.iloc[idx]['ExerciseID']: scores_cb[idx] for idx in df_exercises.index}
        for _, r in candidate_ex.iterrows():
            penalty = abs(r['Difficulty'] - target_diff) * 0.4
            final_list.append({"ex": r.to_dict(), "score": float(dict_cb.get(r['ExerciseID'], 0) - penalty)})
    else:
        for _, r in candidate_ex.iterrows():
            final_list.append({"ex": r.to_dict(), "score": float(1.0 / (abs(r['Difficulty'] - target_diff) + 1.0))})

    sorted_res = sorted(final_list, key=lambda x: x['score'], reverse=True)
    return {
        "status": "success", "avg_score": fixed_avg_score, "academic_rank": academic_rank,
        "subject_score": round(current_comp, 1), "recommendations": [i['ex'] for i in sorted_res[:request.top_k]]
    }

@app.get("/api/history/{student_id}", tags=["Sinh Viên", "Giảng Viên"])
def get_student_history(student_id: int, x_user_role: str = Header(None)):
    if x_user_role not in ["student", "teacher"]: raise HTTPException(status_code=403)
    conn = get_db_connection()
    query = text("SELECT h.MaBaiTap AS ExerciseID, b.TenBaiTap AS Title, h.DiemSo AS Score, ISNULL(b.MaDoKho, 1) AS Difficulty FROM AI_LichSuLamBai h JOIN BAITAP b ON h.MaBaiTap = b.Id WHERE h.MaSinhVien = :sv_id ORDER BY h.DiemSo DESC")
    df = pd.read_sql(query, conn, params={"sv_id": student_id})
    conn.close()
    return {"status": "success", "history": df.to_dict(orient="records")}

class LoginRequest(BaseModel): username: str; password: str
@app.post("/api/login", tags=["Hệ Thống"])
def login_user(request: LoginRequest):
    conn = get_db_connection()
    df = pd.read_sql(text("SELECT VaiTro, MaNguoiDung, HoTen FROM TAIKHOAN WHERE TenDangNhap = :u AND MatKhau = :p"), conn, params={"u": request.username, "p": request.password})
    conn.close()
    if df.empty: return {"status": "error", "message": "Sai thông tin!"}
    return {"status": "success", "role": df.iloc[0]['VaiTro'], "user_id": int(df.iloc[0]['MaNguoiDung']), "full_name": df.iloc[0]['HoTen'], "username": request.username}

class ChangePasswordRequest(BaseModel):
    user_id: int
    old_password: str
    new_password: str

@app.post("/api/change-password", tags=["Hệ Thống"])
def change_password(request: ChangePasswordRequest, x_user_role: str = Header(None)):
    if not x_user_role: raise HTTPException(status_code=401)
    
    conn = get_db_connection()
    try:
        query_check = text("SELECT MatKhau FROM TAIKHOAN WHERE MaNguoiDung = :uid")
        df = pd.read_sql(query_check, conn, params={"uid": request.user_id})
        
        if df.empty: return {"status": "error", "message": "Không tìm thấy tài khoản!"}
            
        current_db_password = str(df.iloc[0]['MatKhau']).strip()
        if current_db_password != request.old_password:
            return {"status": "error", "message": "Mật khẩu cũ không chính xác!"}
            
        query_update = text("UPDATE TAIKHOAN SET MatKhau = :new_pw WHERE MaNguoiDung = :uid")
        conn.execute(query_update, {"new_pw": request.new_password, "uid": request.user_id})
        conn.commit()
        
        return {"status": "success", "message": "Đổi mật khẩu thành công!"}
    except Exception as e:
        return {"status": "error", "message": f"Lỗi hệ thống: {str(e)}"}
    finally:
        conn.close()

@app.get("/api/teacher/overview", tags=["Giảng Viên"])
def get_teacher_overview(x_user_role: str = Header(None)):
    if x_user_role != "teacher": raise HTTPException(status_code=403)
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
def serve_frontend(): return FileResponse("index.html") if os.path.exists("index.html") else {"m": "No index.html"}
