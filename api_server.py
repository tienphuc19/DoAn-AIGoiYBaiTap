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
    
    # 1. GIAO TIẾP VỚI SUPABASE (Chỉ lấy điểm hiển thị và điểm nền)
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

    # 2. XỬ LÝ DỮ LIỆU TỪ SQL SERVER (Cốt lõi của hệ thống)
    df_exercises, df_history = load_data_from_sql()
    df_exercises['Tags'] = df_exercises['Tags'].fillna('')
    if request.subject_code:
        df_exercises = df_exercises[df_exercises['SubjectCode'].str.contains(request.subject_code, case=False, na=False)].reset_index(drop=True)
    
    student_history = df_history[df_history['StudentID'] == request.student_id]
    
    if df_exercises.empty: 
        return {"status": "success", "cf_error_message": "Chưa có dữ liệu bài tập trong SQL.", "avg_score": fixed_avg_score, "academic_rank": academic_rank, "recommendations": []}

    all_attempts_ids = student_history['ExerciseID'].tolist()
    passed_exercises_ids = student_history[student_history['Score'] >= 5.0]['ExerciseID'].tolist()
    
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

    # Lọc bài tập: Chưa làm & Độ khó phù hợp (Không quá sức)
    candidate_exercises = df_exercises[(~df_exercises['ExerciseID'].isin(all_attempts_ids)) & (df_exercises['Difficulty'] <= current_level + 1)]
    if candidate_exercises.empty: 
        return {"status": "success", "cf_error_message": "Tuyệt vời! Bạn đã hoàn thành lộ trình môn này.", "avg_score": fixed_avg_score, "academic_rank": academic_rank, "recommendations": []}

    # 4. THUẬT TOÁN CONTENT-BASED FILTERING (So khớp đặc trưng bài tập)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_exercises['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    passed_indices = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises_ids)].index.tolist()
    
    final_recommendations_with_score = []
    if passed_indices:
        # Tính độ tương đồng với các bài đã giải
        sim_scores_cb = sum([cosine_sim[i] for i in passed_indices])
        if type(sim_scores_cb) != list and sim_scores_cb.max() > sim_scores_cb.min():
            sim_scores_cb = (sim_scores_cb - sim_scores_cb.min()) / (sim_scores_cb.max() - sim_scores_cb.min())
            
        cb_scores_dict = {df_exercises.iloc[idx]['ExerciseID']: sim_scores_cb[idx] for idx in df_exercises.index}
        
        for _, row in candidate_exercises.iterrows():
            cb_score = cb_scores_dict.get(row['ExerciseID'], 0)
            final_recommendations_with_score.append({"exercise": row.to_dict(), "final_score": float(cb_score)})
    else:
        # Nếu chưa làm bài nào, gợi ý ngẫu nhiên hoặc bài dễ trong list Candidate
        for _, row in candidate_exercises.iterrows():
            final_recommendations_with_score.append({"exercise": row.to_dict(), "final_score": float(1.0 / (row['Difficulty'] + 1))}) # Ưu tiên bài dễ

    # Sắp xếp từ điểm phù hợp cao xuống thấp
    sorted_recommendations = sorted(final_recommendations_with_score, key=lambda x: x['final_score'], reverse=True)
    
    return {
        "status": "success", 
        "current_level": int(current_level), 
        "avg_score": float(fixed_avg_score), 
        "academic_rank": academic_rank,
        "cf_error_message": "AI Đang hoạt động (Content-Based Filtering)",
        "recommendations": [item['exercise'] for item in sorted_recommendations[:request.top_k]]
    }
