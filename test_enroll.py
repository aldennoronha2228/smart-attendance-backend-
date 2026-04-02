from fastapi.testclient import TestClient

from main import app


def main() -> None:
    client = TestClient(app)

    with open("sdk/test/1.jpg", "rb") as img1, open("sdk/test/2.png", "rb") as img2:
        response = client.post(
            "/enroll",
            data={"name": "Alden"},
            files=[
                ("images", ("1.jpg", img1, "image/jpeg")),
                ("images", ("2.png", img2, "image/png")),
            ],
        )

    print("status:", response.status_code)
    print("response:", response.json())


if __name__ == "__main__":
    main()
