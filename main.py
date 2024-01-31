from multiprocessing import Process, Queue
from card_detector import detect_cards
from card_stats import view_card_count_window

stream_path = "http://10.128.243.160:8080/video"
video_path = "vids/karty8.mp4"

if __name__ == "__main__":
    shared_queue = Queue()

    p1 = Process(target=view_card_count_window, args=(shared_queue,))
    p2 = Process(target=detect_cards, args=(shared_queue, stream_path))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
