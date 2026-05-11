import pytest
from django.contrib.auth.models import User
from playwright.sync_api import Page, expect


@pytest.fixture(scope="function")
def e2e_user(django_db_setup, db):
    """Tạo user trong Test Database để đăng nhập"""
    user = User.objects.create_user(username="test_student", password="password123")
    return user


@pytest.mark.django_db(transaction=True)  # transaction=True bắt buộc cho live_server
def test_user_submit_flow(live_server, page: Page, e2e_user):
    """
    E2E_01 & E2E_02:
    1. Đăng nhập
    2. Nộp code C++ hợp lệ
    3. Trình duyệt tự chuyển sang trang Result và chạy animation "Dây thun"
    4. Trình duyệt render Heatmap thành công sau khi có kết quả
    """

    # 1. Đăng nhập
    page.goto(f"{live_server.url}/accounts/login/")
    page.fill('input[name="username"]', "test_student")
    page.fill('input[name="password"]', "password123")
    page.click('button[type="submit"]')

    # Đảm bảo đã login thành công và đang ở trang Submit
    expect(page).to_have_url(f"{live_server.url}/submit/")

    # 2. Nhập code C++ vào textarea
    sample_code = """
    #include <iostream>
    using namespace std;
    int main() {
        cout << "Hello AI!" << endl;
        return 0;
    }
    """
    page.fill('textarea[name="code_input"]', sample_code)

    # Nhấn nút Submit
    page.click('button[type="submit"]')

    # 3. Đợi redirect sang trang Result
    page.wait_for_url(f"{live_server.url}/submit/result/*", timeout=10000)

    # 4. Kiểm tra Progress Bar / Spinner
    progress_bar = page.locator('.progress-bar, #progress-bar, .spinner').first
    if progress_bar.is_visible():
        print("Đã phát hiện UI đang tải (Dây thun/Spinner)")

    # 5. Đợi kết quả SSE stream hoàn thành và render Heatmap
    expect(
        page.locator("text=AI GENERATED").or_(page.locator("text=HUMAN WRITTEN"))
    ).to_be_visible(timeout=30000)

    # 6. Kiểm tra các thẻ phân mảnh (Chunk cards)
    chunks = page.locator('.chunk-card, .card')
    assert chunks.count() >= 1, "Phải có ít nhất 1 chunk được phân tích"