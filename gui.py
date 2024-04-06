import threading
import speech_recognition as sr
from ultralytics import YOLO
import cv2
import os
from langchain_community.llms import Ollama


# 음성 인식 설정
r = sr.Recognizer()
mic = sr.Microphone()

# 웹캠 캡처 설정
cap = cv2.VideoCapture(0)

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")

llm = Ollama(model="llama2")

# 물체 인식 스레드 실행 여부 플래그
detect_thread = None
detect_thread_running = False

def listen_and_detect():
    global detect_thread, detect_thread_running
    while True:
        try:
            with mic as source:
                print("말씀해 주세요...")
                audio = r.listen(source)
                text = r.recognize_google(audio, language='ko-KR')
                print(f"인식된 텍스트: {text}")
                if '사진' in text:
                    if not detect_thread_running:
                        detect_thread_running = True
                        detect_thread = threading.Thread(target=detect_objects)
                        detect_thread.start()
                elif '멈춰' in text:
                    detect_thread_running = False
                    if detect_thread is not None:
                        detect_thread.join()
                        detect_thread = None
                elif '끝내기' in text:
                    detect_thread_running = False
                    if detect_thread is not None:
                        detect_thread.join()
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        except sr.UnknownValueError:
            print("음성을 인식하지 못했습니다.")
        except sr.RequestError as e:
            print(f"에러가 발생했습니다; {e}")

def detect_objects():
    global detect_thread_running
    while detect_thread_running:
        # 웹캘이 프레임 읽기
        ret, frame = cap.read()
        # YOLOv8 모델로 물체 인식
        results = model(frame)
        # 인식된 물체 라벨 가져오기
        labels = [model.names[int(label)] for label in results[0].boxes.cls]
        # LLaMa 2 API에 질문하고 대답 받기
        response = ask_llm(labels)
        print(f"LLaMa 2 응답: {response}")
        # 인식된 물체 위치 표시
        annotated_frame = results[0].plot()
        cv2.imshow('Object Detection', annotated_frame)
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detect_thread_running = False
            break
    cv2.destroyAllWindows()

def ask_llm(labels):
    prompt = {'role': 'user', 'content': f"다음 물체가 인식되었습니다: {', '.join(labels)}. 이 물체들에 대해 설명해주세요."}
    response = llm.invoke([prompt])
    return response.strip()

if __name__ == "__main__":
    listen_thread = threading.Thread(target=listen_and_detect)
    listen_thread.daemon = True
    listen_thread.start()
    listen_thread.join()