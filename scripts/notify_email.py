"""
Send an email notification once a training/inference run finishes, so it can be
started detached on the clinic PC without needing to stay logged in.

Reads SMTP credentials from environment variables (kept out of the repo/image):
    SMTP_HOST       (default: mail.gmx.net)
    SMTP_PORT       (default: 587, STARTTLS)
    SMTP_USER       GMX address, e.g. yourname@gmx.de
    SMTP_PASSWORD   GMX app password (Sicherheit -> App-Passwoerter), NOT the
                    normal login password
    NOTIFY_EMAIL_TO recipient address (defaults to SMTP_USER if unset)

If SMTP_USER/SMTP_PASSWORD are not set, this prints a warning and exits
successfully (0) rather than failing the run — a missing notification should
never mask the actual training exit code.

Usage:
    python scripts/notify_email.py --exit-code 0 --log /workspace/training.log
"""

import argparse
import os
import smtplib
import sys
from email.mime.text import MIMEText

MAX_LOG_CHARS = 20_000  # keep the email body reasonably sized


def tail(text, max_chars):
    return text[-max_chars:] if len(text) > max_chars else text


def extract_summary(log_text):
    """
    Pull out the most relevant lines (loss/metric values, checkpoint saves,
    errors) so the email body is a short summary rather than the full log.
    """
    keywords = (
        "train_loss", "Val loss", "Best checkpoint", "Checkpoint saved",
        "Epoch", "Error", "Traceback", "CUDA out of memory", "Killed",
    )
    lines = [ln for ln in log_text.splitlines() if any(k in ln for k in keywords)]
    return "\n".join(lines[-200:])  # cap to the last 200 matching lines


def main():
    parser = argparse.ArgumentParser(description="Email a training run summary.")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--log", required=True, help="Path to the captured training log.")
    args = parser.parse_args()

    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        print("notify_email.py: SMTP_USER/SMTP_PASSWORD not set — skipping email notification.")
        return  # do not fail the run just because notification isn't configured

    smtp_host = os.environ.get("SMTP_HOST", "mail.gmx.net")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    recipient = os.environ.get("NOTIFY_EMAIL_TO", smtp_user)

    log_text = ""
    if os.path.exists(args.log):
        with open(args.log, "r", errors="replace") as f:
            log_text = f.read()

    status = "SUCCESS" if args.exit_code == 0 else f"FAILED (exit code {args.exit_code})"
    summary = extract_summary(log_text) or "(no matching summary lines found)"
    tail_raw = tail(log_text, MAX_LOG_CHARS)

    body = (
        f"Training run finished: {status}\n\n"
        f"---- Summary (matched lines) ----\n{summary}\n\n"
        f"---- Log tail (last {MAX_LOG_CHARS} chars) ----\n{tail_raw}\n"
    )

    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = f"[ophthalmic-diffusion-classifier] Training {status}"
    msg["From"] = smtp_user
    msg["To"] = recipient

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [recipient], msg.as_string())
        print(f"notify_email.py: notification sent to {recipient}")
    except Exception as exc:
        # Never let a notification failure mask the real training exit code.
        print(f"notify_email.py: failed to send email: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
