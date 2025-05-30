from flask import jsonify, send_file

from src.core.context import TranscriptionContext


def register_routes(app, transcription_context: TranscriptionContext):
    @app.route('/')
    def serve_html():
        return send_file('../static/index.html')

    @app.route('/start_recording', methods=['POST'])
    def start_recording():
        try:
            transcription_context.start()
            return jsonify({"message": "Recording started"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/stop_recording', methods=['POST'])
    def stop_recording():
        transcription_context.stop()
        return jsonify({"message": "Recording stopped"})
