import sys
import random
import time
from pathlib import Path

from selenium.common.exceptions import (
    InvalidSessionIdException,
    NoSuchWindowException,
)
from selenium.webdriver.common.by import By

local_python_path = str(Path(__file__).parents[1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger

logger = get_logger(__name__)
config = load_config(add_date=False, config_path=Path(local_python_path) / "config.json")
CAPTCHA_WAIT_MESSAGE = (
    "This website uses a security service to protect against malicious bots. "
    "This page is displayed while the website verifies you are not a bot."
)


CHROME_UNAVAILABLE_MESSAGE_PARTS = (
    "chrome not reachable",
    "disconnected: not connected to devtools",
    "failed to check if window was closed",
    "invalid session id",
    "session deleted because of page crash",
    "tab crashed",
    "target window already closed",
    "unable to receive message from renderer",
    "web view not found",
)


def exception_indicates_chrome_unavailable(error):
    if isinstance(error, (InvalidSessionIdException, NoSuchWindowException)):
        return True
    error_text = f"{type(error).__name__}: {error}".lower()
    return any(
        message_part in error_text
        for message_part in CHROME_UNAVAILABLE_MESSAGE_PARTS
    )


def is_chrome_unavailable_error(driver, error):
    if exception_indicates_chrome_unavailable(error):
        return True
    try:
        driver.current_window_handle
    except Exception:
        return True
    return False


def get_chrome_restart_block_reason(
    restart_count,
    scraped_since_restart,
    max_restarts,
):
    if restart_count >= max_restarts:
        return f"Chrome has already been restarted {max_restarts} times"
    if restart_count > 0 and scraped_since_restart == 0:
        return "nothing new was scraped since the previous Chrome restart"
    return ""


def quit_driver(driver):
    if driver is None:
        return
    try:
        driver.quit()
    except Exception as error:
        logger.warning(f"Could not close the browser cleanly: {error}")


def extract_titles(driver):
    h2_titles = [x.text.strip() for x in driver.find_elements(By.TAG_NAME, "h2") if x.text.strip()]
    return "".join(h2_titles)


def extract_text(driver):
    return "\n".join([x.text for x in driver.find_elements(By.TAG_NAME, "p") if x.text.strip()])


def normalize_whitespace(text):
    return " ".join(str(text).split())


def is_security_service_wait_message(text):
    return normalize_whitespace(text) == CAPTCHA_WAIT_MESSAGE


def is_cloudflare_challenge(driver):
    page_text = driver.page_source.lower()
    if "cloudflare" in page_text and ("verify you are human" in page_text or "attention required" in page_text):
        return True
    body_elements = driver.find_elements(By.TAG_NAME, "body")
    if body_elements:
        body_text = normalize_whitespace(body_elements[0].text)
        if is_security_service_wait_message(body_text):
            return True
    challenge_iframes = driver.find_elements(By.CSS_SELECTOR, "iframe[src*='challenges.cloudflare.com']")
    return len(challenge_iframes) > 0


def wait_for_page_segments(driver, initial_wait_seconds=20, retry_wait_seconds=10, max_total_wait_seconds=180):
    logger.info(
        f"Waiting for <p> segments (initial={initial_wait_seconds}s, "
        f"retry={retry_wait_seconds}s, max={max_total_wait_seconds}s)"
    )
    waited_seconds = 0
    time.sleep(initial_wait_seconds)
    waited_seconds += initial_wait_seconds

    while True:
        extracted_text = extract_text(driver)
        text_ready = bool(extracted_text.strip())
        security_wait_text_active = is_security_service_wait_message(extracted_text)
        challenge_active = is_cloudflare_challenge(driver)
        if text_ready and not challenge_active and not security_wait_text_active:
            break

        if waited_seconds >= max_total_wait_seconds:
            if challenge_active:
                raise TimeoutError(
                    f"Cloudflare challenge still active after waiting {waited_seconds}s "
                    f"(max {max_total_wait_seconds}s)."
                )
            if security_wait_text_active:
                raise TimeoutError(
                    f"Security-service wait text still active after waiting {waited_seconds}s "
                    f"(max {max_total_wait_seconds}s)."
                )
            raise TimeoutError(
                f"No non-empty <p> text found after waiting {waited_seconds}s "
                f"(max {max_total_wait_seconds}s)."
            )

        next_wait_seconds = min(retry_wait_seconds, max_total_wait_seconds - waited_seconds)
        if challenge_active:
            logger.info(
                f"Cloudflare challenge still active after {waited_seconds}s. "
                f"Waiting {next_wait_seconds}s more."
            )
        elif security_wait_text_active:
            logger.info(
                f"Security-service wait text still active after {waited_seconds}s. "
                f"Waiting {next_wait_seconds}s more."
            )
        else:
            logger.info(
                f"No non-empty <p> text yet after {waited_seconds}s. "
                f"Waiting {next_wait_seconds}s more."
            )
        time.sleep(next_wait_seconds)
        waited_seconds += next_wait_seconds

    logger.info(f"Page segments found after {waited_seconds}s")


def handle_cloudflare_if_needed(driver, url):
    if not is_cloudflare_challenge(driver):
        return
    logger.warning("Cloudflare challenge detected; waiting for manual solve")
    input(
        "\nCloudflare challenge detected in browser.\n"
        "Please solve it manually, then press Enter here to continue..."
    )
    driver.get(url)
    logger.info("Continuing after manual Cloudflare confirmation")


def wait_for_cdnc_content_ready(
    driver,
    url,
    pre_wait_seconds=10,
    initial_wait_seconds=20,
    retry_wait_seconds=10,
    max_total_wait_seconds=300,
):
    logger.info(
        f"Preparing page for extraction (pre_wait={pre_wait_seconds}s, "
        f"initial={initial_wait_seconds}s, retry={retry_wait_seconds}s, "
        f"max={max_total_wait_seconds}s)"
    )
    if pre_wait_seconds > 0:
        time.sleep(pre_wait_seconds)
    handle_cloudflare_if_needed(driver, url)
    wait_for_page_segments(
        driver,
        initial_wait_seconds=initial_wait_seconds,
        retry_wait_seconds=retry_wait_seconds,
        max_total_wait_seconds=max_total_wait_seconds,
    )


def wait_for_searchresults_ready(
    driver,
    url,
    pre_wait_seconds=10,
    retry_wait_seconds=10,
    max_total_wait_seconds=300,
):
    logger.info(
        f"Preparing search page (pre_wait={pre_wait_seconds}s, "
        f"retry={retry_wait_seconds}s, max={max_total_wait_seconds}s)"
    )
    if pre_wait_seconds > 0:
        time.sleep(pre_wait_seconds)
    handle_cloudflare_if_needed(driver, url)

    waited_seconds = 0
    while True:
        challenge_active = is_cloudflare_challenge(driver)
        search_results_elements = driver.find_elements(By.CLASS_NAME, "searchresults")
        has_search_results = len(search_results_elements) > 0
        if has_search_results and not challenge_active:
            logger.info(f"Search results container found after {waited_seconds}s")
            return

        if waited_seconds >= max_total_wait_seconds:
            if challenge_active:
                raise TimeoutError(
                    f"Cloudflare/security challenge still active after waiting {waited_seconds}s "
                    f"(max {max_total_wait_seconds}s)."
                )
            raise TimeoutError(
                f"No searchresults container found after waiting {waited_seconds}s "
                f"(max {max_total_wait_seconds}s)."
            )

        next_wait_seconds = min(retry_wait_seconds, max_total_wait_seconds - waited_seconds)
        if challenge_active:
            logger.info(
                f"Security/challenge still active after {waited_seconds}s. "
                f"Waiting {next_wait_seconds}s more."
            )
        else:
            logger.info(
                f"No searchresults container yet after {waited_seconds}s. "
                f"Waiting {next_wait_seconds}s more."
            )
        time.sleep(next_wait_seconds)
        waited_seconds += next_wait_seconds


def sleep_between_fetches(min_sleep_seconds, max_sleep_seconds):
    sleep_seconds = random.uniform(min_sleep_seconds, max_sleep_seconds)
    logger.info(f"Sleeping {sleep_seconds:.1f}s before next fetch")
    time.sleep(sleep_seconds)
