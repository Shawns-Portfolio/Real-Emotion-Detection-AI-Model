from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import os
from threading import Thread
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()
# Allow frontend access (important for JavaScript)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/process-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        print("Received image for processing.")
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")
        image = image.resize((96, 96))
        response = run_func(image)
        print("something something")

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)



# variable declaration
img_size = 96
anger = {'emotion1':None, 'confidence1':0,'lies':False, 'emotion2':None, 'confidence2':0}
contempt = {'emotion1':None, 'confidence1':0,'lies':False, 'emotion2':None, 'confidence2':0}
disgust = {'emotion1':None, 'confidence1':0,'lies':False, 'emotion2':None, 'confidence2':0}
fear = {'emotion1':None, 'confidence1':0,'lies':False, 'emotion2':None, 'confidence2':0}
sad = {'emotion1':None, 'confidence1':0,'lies':False, 'emotion2':None, 'confidence2':0}
surprise = {'emotion1':None, 'confidence1':0,'lies':False, 'emotion2':None, 'confidence2':0}



# First layer, decision making

#anger
anger_model1 = load_model('models/anger_happy/model.keras')
anger_model2 = load_model('models/anger_neutral/model.keras')
def anger_f(img):
    global anger
    #img1 = image.load_img(img, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis = 0)
    #predict
    pred1 = anger_model1.predict(img_array)
    pred2 = anger_model2.predict(img_array)
    index1 = np.argmax(pred1)
    index2 = np.argmax(pred2)
    confidence1 = np.max(pred1)
    confidence2 = np.max(pred2)

    #this only gives anger if both models says anger
    if index1 == index2:
        if index1 == 0:
            overall_confidence = (float(confidence1) + float(confidence2))/2
            anger['emotion1'] = "anger"
            anger['confidence1'] = overall_confidence
            anger['lies'] = False
            return
        elif index2 == 1: 
            anger['emotion1'] = 'happy'
            anger['confidence1'] = confidence1
            anger['lies'] = True
            anger['emotion2'] = 'neutral'
            anger['confidence2'] = confidence2
            return
    else:
        if index1 == 1:
            anger['emotion1'] = 'happy'
            anger['confidence1'] = confidence1
        else:
            anger['emotion1'] = 'neutral'
            anger['confidence1'] = confidence2
        anger['lies'] = False
        return

#contempt
contempt_model1 = load_model('models/contempt_happy/model.keras')
contempt_model2 = load_model('models/contempt_neutral/model.keras')
def contempt_f(img):
    global contempt
    #img1 = image.load_img(img, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis = 0)
    #predict
    pred1 = contempt_model1.predict(img_array)
    pred2 = contempt_model2.predict(img_array)
    index1 = np.argmax(pred1)
    index2 = np.argmax(pred2)
    confidence1 = np.max(pred1)
    confidence2 = np.max(pred2)

    #this only gives anger if both models says anger
    if index1 == index2:
        if index1 == 0:
            overall_confidence = (float(confidence1) + float(confidence2))/2
            contempt['emotion1'] = "contempt"
            contempt['confidence1'] = overall_confidence
            contempt['lies'] = False
            return
        elif index2 == 1: 
            contempt['emotion1'] = 'happy'
            contempt['confidence1'] = confidence1
            contempt['lies'] = True
            contempt['emotion2'] = 'neutral'
            contempt['confidence2'] = confidence2
            return
    else:
        if index1 == 1:
            contempt['emotion1'] = 'happy'
            contempt['confidence1'] = confidence1
        else:
            contempt['emotion1'] = 'neutral'
            contempt['confidence1'] = confidence2
        contempt['lies'] = False
        return
    
#fear
fear_model1 = load_model('models/fear_happy/model.keras')
fear_model2 = load_model('models/fear_neutral/model.keras')
def fear_f(img):
    global contempt
    #img1 = image.load_img(img, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis = 0)
    #predict
    pred1 = fear_model1.predict(img_array)
    pred2 = fear_model2.predict(img_array)
    index1 = np.argmax(pred1)
    index2 = np.argmax(pred2)
    confidence1 = np.max(pred1)
    confidence2 = np.max(pred2)

    #this only gives anger if both models says anger
    if index1 == index2:
        if index1 == 0:
            overall_confidence = (float(confidence1) + float(confidence2))/2
            fear['emotion1'] = "fear"
            fear['confidence1'] = overall_confidence
            fear['lies'] = False
            return
        elif index2 == 1: 
            fear['emotion1'] = 'happy'
            fear['confidence1'] = confidence1
            fear['lies'] = True
            fear['emotion2'] = 'neutral'
            fear['confidence2'] = confidence2
            return
    else:
        if index1 == 1:
            fear['emotion1'] = 'happy'
            fear['confidence1'] = confidence1
        else:
            fear['emotion1'] = 'neutral'
            fear['confidence1'] = confidence2
        contempt['lies'] = False
        return

#disgust
disgust_model1 = load_model('models/disgust_happy/model.keras')
disgust_model2 = load_model('models/disgust_neutral/model.keras')
def disgust_f(img):
    global disgust
    #img1 = image.load_img(img, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis = 0)
    #predict
    pred1 = disgust_model1.predict(img_array)
    pred2 = disgust_model2.predict(img_array)
    index1 = np.argmax(pred1)
    index2 = np.argmax(pred2)
    confidence1 = np.max(pred1)
    confidence2 = np.max(pred2)

    #this only gives anger if both models says anger
    if index1 == index2:
        if index1 == 0:
            overall_confidence = (float(confidence1) + float(confidence2))/2
            disgust['emotion1'] = "disgust"
            disgust['confidence1'] = overall_confidence
            disgust['lies'] = False
            return
        elif index2 == 1: 
            disgust['emotion1'] = 'happy'
            disgust['confidence1'] = confidence1
            disgust['lies'] = True
            disgust['emotion2'] = 'neutral'
            disgust['confidence2'] = confidence2
            return
    else:
        if index1 == 1:
            disgust['emotion1'] = 'happy'
            disgust['confidence1'] = confidence1
        else:
            disgust['emotion1'] = 'neutral'
            disgust['confidence1'] = confidence2
        disgust['lies'] = False
        return


#sad
sad_model1 = load_model('models/happy_sad/model.keras')
sad_model2 = load_model('models/neutral_sad/model.keras')
def sad_f(img):
    global disgust
    #img1 = image.load_img(img, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis = 0)
    #predict
    pred1 = sad_model1.predict(img_array)
    pred2 = sad_model2.predict(img_array)
    index1 = np.argmax(pred1)
    index2 = np.argmax(pred2)
    confidence1 = np.max(pred1)
    confidence2 = np.max(pred2)

    #this only gives anger if both models says anger
    if index1 == index2:
        if index1 == 1:
            overall_confidence = (float(confidence1) + float(confidence2))/2
            sad['emotion1'] = "sad"
            sad['confidence1'] = overall_confidence
            sad['lies'] = False
            return
        elif index2 == 0: 
            sad['emotion1'] = 'happy'
            sad['confidence1'] = confidence1
            sad['lies'] = True
            sad['emotion2'] = 'neutral'
            sad['confidence2'] = confidence2
            return
    else:
        if index1 == 0:
            sad['emotion1'] = 'happy'
            sad['confidence1'] = confidence1
        else:
            sad['emotion1'] = 'neutral'
            sad['confidence1'] = confidence2
        disgust['lies'] = False
        return

# surprise
surprise_model1 = load_model('models/happy_surprise/model.keras')
surprise_model2 = load_model('models/neutral_surprise/model.keras')
def surprise_f(img):
    global disgust
    #img1 = image.load_img(img, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis = 0)
    #predict
    pred1 = surprise_model1.predict(img_array)
    pred2 = surprise_model2.predict(img_array)
    index1 = np.argmax(pred1)
    index2 = np.argmax(pred2)
    confidence1 = np.max(pred1)
    confidence2 = np.max(pred2)

    #this only gives anger if both models says anger
    if index1 == index2:
        if index1 == 1:
            overall_confidence = (float(confidence1) + float(confidence2))/2
            surprise['emotion1'] = "surprise"
            surprise['confidence1'] = overall_confidence
            surprise['lies'] = False
            return
        elif index2 == 0: 
            surprise['emotion1'] = 'happy'
            surprise['confidence1'] = confidence1
            surprise['lies'] = True
            surprise['emotion2'] = 'neutral'
            surprise['confidence2'] = confidence2
            return
    else:
        if index1 == 0:
            surprise['emotion1'] = 'happy'
            surprise['confidence1'] = confidence1
        else:
            surprise['emotion1'] = 'neutral'
            surprise['confidence1'] = confidence2
        disgust['lies'] = False
        return





def run_func(img):
    #create threads
    anger_thread = Thread(target=anger_f, args=(img,))
    contempt_thread = Thread(target=contempt_f, args=(img,))
    disgust_thread = Thread(target=disgust_f, args=(img,))
    fear_thread = Thread(target=fear_f, args=(img,))
    sad_thread = Thread(target=sad_f, args=(img,)) 
    surprise_thread = Thread(target=surprise_f, args=(img,))
    #start threads
    anger_thread.start()
    contempt_thread.start() 
    disgust_thread.start()
    fear_thread.start()
    sad_thread.start()
    surprise_thread.start()
    #join threads
    anger_thread.join()
    contempt_thread.join()
    disgust_thread.join()
    fear_thread.join()
    sad_thread.join()
    surprise_thread.join()
    '''print(contempt)
    print(anger)
    print(disgust)
    print(fear)
    print(sad)
    print(surprise)
    '''
    pool = {'contempt':0, 'anger':0, 'disgust':0, 'fear':0, 'sad':0, 'surprise':0, 'happy':0, 'neutral':0}
   # picking only emotions with highest confidence
    if pool[f'{contempt["emotion1"]}'] < contempt['confidence1']:
        pool[f'{contempt["emotion1"]}'] = contempt['confidence1']
    if contempt['lies']:
        if pool[f'{contempt["emotion2"]}'] < contempt['confidence2']:
            pool[f'{contempt["emotion2"]}'] = contempt['confidence2']

    if pool[f'{anger["emotion1"]}'] < anger['confidence1']:
        pool[f'{anger["emotion1"]}'] = anger['confidence1']
    if contempt['lies']:
        if pool[f'{anger["emotion2"]}'] < anger['confidence2']:
            pool[f'{anger["emotion2"]}'] = anger['confidence2']

    if pool[f'{disgust["emotion1"]}'] < disgust['confidence1']:
        pool[f'{disgust["emotion1"]}'] = disgust['confidence1']
    if disgust['lies']:
        if pool[f'{disgust["emotion2"]}'] < disgust['confidence2']:
            pool[f'{disgust["emotion2"]}'] = disgust['confidence2']

    if pool[f'{fear["emotion1"]}'] < fear['confidence1']:
        pool[f'{fear["emotion1"]}'] = fear['confidence1']
    if fear['lies']:
        if pool[f'{fear["emotion2"]}'] < fear['confidence2']:
            pool[f'{fear["emotion2"]}'] = fear['confidence2']

    if pool[f'{sad["emotion1"]}'] < sad['confidence1']:
        pool[f'{sad["emotion1"]}'] = sad['confidence1']
    if sad['lies']:
        if pool[f'{sad["emotion2"]}'] < sad['confidence2']:
            pool[f'{sad["emotion2"]}'] = sad['confidence2']

    if pool[f'{surprise["emotion1"]}'] < surprise['confidence1']:
        pool[f'{surprise["emotion1"]}'] = surprise['confidence1']
    if surprise['lies']:
        if pool[f'{surprise["emotion2"]}'] < surprise['confidence2']:
            pool[f'{surprise["emotion2"]}'] = surprise['confidence2']
    print("Final Pool Results:")
    print(pool)
    #determine the final emotion
    final_emotion = ['',0]
    final_emotion[0] = 'anger'
    final_emotion[1] = pool['anger']
    if pool['contempt'] > final_emotion[1]:
        final_emotion[0] = 'contempt'
        final_emotion[1] = pool['contempt']
    if pool['disgust'] > final_emotion[1]:
        final_emotion[0] = 'disgust'
        final_emotion[1] = pool['disgust']
    if pool['fear'] > final_emotion[1]:
        final_emotion[0] = 'fear'
        final_emotion[1] = pool['fear']
    if pool['sad'] > final_emotion[1]:
        final_emotion[0] = 'sad'
        final_emotion[1] = pool['sad']
    if pool['surprise'] > final_emotion[1]:
        final_emotion[0] = 'surprise'
        final_emotion[1] = pool['surprise']
    if final_emotion[1] == 0 or final_emotion[1] < 0.95:
        final_emotion[0] = 'neutral'
        final_emotion[1] = pool['neutral']
        if pool['happy'] > final_emotion[1]:
            final_emotion[0] = 'happy'
            final_emotion[1] = pool['happy']

    final_emotion[1] = int(final_emotion[1]*100)
    print(f"Final Emotion: {final_emotion[0]} with confidence {final_emotion[1]}")
    return final_emotion
    

'''
dir_path = 'sample'

for img_file in os.listdir(dir_path):
    img_path = os.path.join(dir_path, img_file)
    print(f"Processing image: {img_file}")
    run_func(img_path)
    a = input("Press Enter to continue to the next image...")
'''