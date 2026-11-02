"""Arabic presentation of provisional per-leaf results; no accuracy inference."""
PLANTS={"Apple":"تفاح","Blueberry":"توت أزرق","Cherry_(including_sour)":"كرز","Corn_(maize)":"ذرة","Grape":"عنب","Orange":"برتقال","Peach":"خوخ","Pepper,_bell":"فلفل حلو","Potato":"بطاطس","Raspberry":"توت العليق","Soybean":"فول الصويا","Squash":"قرع","Strawberry":"فراولة","Tomato":"طماطم"}
DISEASES={"healthy":"سليم بحسب التوقع","Apple_scab":"جرب التفاح","Black_rot":"العفن الأسود","Cedar_apple_rust":"صدأ التفاح","Powdery_mildew":"البياض الدقيقي","Cercospora_leaf_spot Gray_leaf_spot":"تبقع الأوراق الرمادي","Common_rust_":"الصدأ الشائع","Northern_Leaf_Blight":"لفحة الأوراق الشمالية","Esca_(Black_Measles)":"إسكا / الحصبة السوداء","Leaf_blight_(Isariopsis_Leaf_Spot)":"لفحة وتبقع أوراق العنب","Haunglongbing_(Citrus_greening)":"اخضرار الحمضيات","Bacterial_spot":"التبقع البكتيري","Early_blight":"اللفحة المبكرة","Late_blight":"اللفحة المتأخرة","Leaf_scorch":"احتراق الأوراق","Leaf_Mold":"عفن الأوراق","Septoria_leaf_spot":"تبقع سبتوريا","Spider_mites Two-spotted_spider_mite":"العنكبوت الأحمر ذو البقعتين","Target_Spot":"التبقع الهدفّي","Tomato_Yellow_Leaf_Curl_Virus":"فيروس تجعد واصفرار أوراق الطماطم","Tomato_mosaic_virus":"فيروس موزاييك الطماطم"}
FLAGS={"low_confidence":"ثقة منخفضة","raw_crop_disagreement":"اختلاف توقع الورقة عن الصورة الكاملة","small_object_candidate":"منطقة صغيرة","tomato_shift_candidate":"اختلاف نوع النبات بين الورقة والصورة"}

def class_label(name):
    plant,_,disease=name.partition('___')
    return PLANTS.get(plant,plant.replace('_',' ')),DISEASES.get(disease,disease.replace('_',' '))

def present(result):
    gate=result['plant_gate']['label'];count=result['detected_leaves']
    if gate=='not_plant':
        status='لم تتعرف بوابة النموذج على نبات في الصورة، لذلك توقف التحليل. قد تخطئ البوابة؛ جرّب صورة أقرب وأوضح للأوراق.'
    elif not count:
        status='مرّت الصورة ببوابة النبات، لكن الكاشف لم يجد مناطق أوراق. جرّب صورة أقرب وأوضح.'
    elif gate=='uncertain':
        status='بوابة النبات غير متأكدة؛ جميع التوقعات التالية تحتاج مراجعة.'
    else:status='اكتمل التحليل التجريبي. كل منطقة تمثل ورقة محتملة، والتوقعات تحتاج تحققًا.'
    summary=f"### {status}\n\n**أوراق محتملة: {count}** · زمن التحليل: {result['timing']['pipeline_seconds']:.1f} ثانية\n\nالثقة المعروضة هي ثقة النموذج في توقعه، وليست نسبة صحة التشخيص."
    rows=[];top3=[];gallery=[]
    for leaf in result['leaves']:
        top=leaf['crop']['top3'][0];plant,disease=class_label(top['class'])
        reasons=[FLAGS.get(flag,flag) for flag in leaf.get('review_flags',[])]
        if gate=='uncertain':reasons.insert(0,'بوابة النبات غير متأكدة')
        review='؛ '.join(reasons) or 'توقع غير مؤكد؛ يحتاج مرجعًا'
        rows.append([leaf['id'],plant,disease,f"{top['confidence']:.1%}",review])
        for rank,pred in enumerate(leaf['crop']['top3'],1):
            species,condition=class_label(pred['class']);top3.append([leaf['id'],rank,species,condition,f"{pred['confidence']:.1%}"])
        crop=leaf['padding_variants'][0].get('saved_crop')
        if crop:gallery.append((crop,f"المنطقة {leaf['id']} — {plant}: {disease} ({top['confidence']:.1%})"))
    return summary,rows,gallery,top3
