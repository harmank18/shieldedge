"""
QR Code Threat Scanner for ShieldEdge.
Decodes QR codes from image files and pipes detected URLs through
the URL phishing detector — all fully on-device, zero network calls.

Usage:
  python3 shieldedge.py --qr /path/to/qrcode.png

Requires: opencv-python-headless, pyzbar, libzbar0
Install:  pip install opencv-python-headless pyzbar
"""

import os
import sys
from typing import Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False


def decode_qr_from_image(image_path: str) -> list:
    """
    Decode all QR codes from an image file.
    Returns list of decoded data strings.
    """
    if not CV2_AVAILABLE:
        raise ImportError(
            "opencv-python-headless not installed. Run: pip install opencv-python-headless"
        )
    if not PYZBAR_AVAILABLE:
        raise ImportError(
            "pyzbar not installed. Run: pip install pyzbar && sudo apt-get install libzbar0"
        )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}. Ensure it is a valid PNG/JPG.")

    # Try multiple preprocessing approaches to maximize QR decode success
    decoded_data = []
    
    # Attempt 1: direct decode
    codes = pyzbar.decode(img)
    
    # Attempt 2: grayscale if no codes found
    if not codes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        codes = pyzbar.decode(gray)
    
    # Attempt 3: threshold + denoise
    if not codes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        codes = pyzbar.decode(thresh)

    for code in codes:
        data = code.data.decode("utf-8", errors="replace").strip()
        code_type = code.type  # QR, QRCODE, etc.
        decoded_data.append({"data": data, "type": code_type})

    return decoded_data


def generate_test_qr(url: str, output_path: str) -> bool:
    """
    Generate a test QR code image for a given URL.
    Requires: pip install qrcode[pil]
    """
    try:
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(output_path)
        return True
    except ImportError:
        return False


class QRScanner:
    """
    Wraps QR code decoding + URL threat detection pipeline.
    """

    def __init__(self, url_detector=None):
        """
        Args:
            url_detector: URLDetector instance. If None, creates one with default model path.
        """
        self.url_detector = url_detector
        if self.url_detector is None:
            from shieldedge.url_detector import URLDetector
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base, "models", "url_model.joblib")
            self.url_detector = URLDetector(model_path)

    def scan(self, image_path: str) -> dict:
        """
        Decode QR code from image and analyze any URLs found.
        
        Returns:
            {
                "image_path": str,
                "qr_codes_found": int,
                "results": [ { "raw_data", "type", "is_url", "url_analysis" } ]
                "overall_verdict": "PHISHING" | "SUSPICIOUS" | "SAFE" | "NO_QR_FOUND"
            }
        """
        result = {
            "image_path": image_path,
            "qr_codes_found": 0,
            "results": [],
            "overall_verdict": "NO_QR_FOUND",
        }

        if not CV2_AVAILABLE or not PYZBAR_AVAILABLE:
            result["error"] = (
                "QR scanning libraries not installed. "
                "Run: pip install opencv-python-headless pyzbar && sudo apt-get install libzbar0"
            )
            return result

        try:
            decoded = decode_qr_from_image(image_path)
        except Exception as e:
            result["error"] = str(e)
            return result

        result["qr_codes_found"] = len(decoded)

        if not decoded:
            result["overall_verdict"] = "NO_QR_FOUND"
            return result

        worst_verdict = "SAFE"
        verdict_priority = {"PHISHING": 3, "SUSPICIOUS": 2, "SAFE": 1, "NO_QR_FOUND": 0}

        for item in decoded:
            data = item["data"]
            is_url = data.startswith(("http://", "https://", "www."))

            entry = {
                "raw_data": data,
                "type": item["type"],
                "is_url": is_url,
            }

            if is_url:
                analysis = self.url_detector.predict(data)
                entry["url_analysis"] = {
                    "verdict": analysis["verdict"],
                    "confidence": analysis["confidence"],
                    "signals": analysis["signals"],
                }
                # Track worst verdict
                v = analysis["verdict"]
                if verdict_priority.get(v, 0) > verdict_priority.get(worst_verdict, 0):
                    worst_verdict = v
            else:
                entry["url_analysis"] = None
                # Non-URL QR (contact, text, etc.) — treat as safe
            
            result["results"].append(entry)

        result["overall_verdict"] = worst_verdict
        return result


def print_scan_result(scan_result: dict):
    """Pretty-print QR scan results to terminal."""
    img = scan_result.get("image_path", "unknown")
    n = scan_result.get("qr_codes_found", 0)
    verdict = scan_result.get("overall_verdict", "UNKNOWN")

    print("\n" + "=" * 60)
    print(f"QR CODE SCAN RESULTS")
    print(f"Image:          {img}")
    print(f"QR Codes Found: {n}")

    if "error" in scan_result:
        print(f"Error:          {scan_result['error']}")
        print("=" * 60)
        return

    color = {
        "PHISHING":    "\033[91m",
        "SUSPICIOUS":  "\033[93m",
        "SAFE":        "\033[92m",
        "NO_QR_FOUND": "\033[94m",
    }.get(verdict, "")
    reset = "\033[0m"
    print(f"Overall Verdict: {color}{verdict}{reset}")
    print("=" * 60)

    for i, r in enumerate(scan_result.get("results", []), 1):
        print(f"\n[QR Code #{i}]")
        print(f"  Type:  {r.get('type', 'UNKNOWN')}")
        print(f"  Data:  {r.get('raw_data', '')[:80]}")
        if r.get("is_url") and r.get("url_analysis"):
            ua = r["url_analysis"]
            v = ua["verdict"]
            c = {
                "PHISHING":   "\033[91m",
                "SUSPICIOUS": "\033[93m",
                "SAFE":       "\033[92m",
            }.get(v, "")
            print(f"  URL Verdict:  {c}{v}{reset} (confidence: {ua['confidence']})")
            for sig in ua.get("signals", []):
                print(f"    • {sig}")
        elif not r.get("is_url"):
            print("  Content: Not a URL — likely contact/text QR code")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m shieldedge.qr_scanner <image_path>")
        sys.exit(1)
    scanner = QRScanner()
    result = scanner.scan(sys.argv[1])
    print_scan_result(result)
