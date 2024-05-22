from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import subprocess
from sys import argv

def out(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    if result.stderr: logging.error(result.stderr)
    return result.stdout

class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith('/q/'):
            self.send_response(404)
            self.end_headers()
            return
        path = self.path[3:]
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        output = out(subprocess.list2cmdline(argv[1:] + [path]))
        self.wfile.write(bytes(output, encoding='utf-8'))

def serveHTTP(port=80):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = HTTPServer(server_address, Server)
    logging.info(f'Starting httpd on :{port}...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    serveHTTP()