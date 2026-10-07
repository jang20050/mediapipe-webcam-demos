"""제스처 리액션 웹 버전 실행.

실행: python run_web.py

1) 최신 학습 모델을 docs/model.json으로 변환
2) docs(웹) 폴더를 로컬 서버로 띄움 (웹캠은 file:// 에서는 동작하지 않아 서버가 필요)
3) 브라우저 자동 열기  ->  [카메라 시작] 클릭
종료: Ctrl+C
"""
import functools
import http.server
import webbrowser

from export_web_model import WEB_DIR, export

PORT = 8000


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # 요청마다 찍히는 로그 숨김
        pass


def main():
    export()
    handler = functools.partial(QuietHandler, directory=str(WEB_DIR))
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    except OSError:  # 8000번이 사용 중이면 빈 포트 사용
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)

    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"웹 서버 실행 중: {url}   (종료: Ctrl+C)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")


if __name__ == "__main__":
    main()
