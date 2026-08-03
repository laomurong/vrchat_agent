import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, 'src')

# ---- 1) 直接 pydantic schema 验证 ----
from vrchat_agent.config import AppConfig
s = AppConfig.model_json_schema()
for cls, kind in [("AudioInputConfig", "input"), ("AudioOutputConfig", "output")]:
    dev = s["$defs"][cls]["properties"]["device"]
    print(f"{cls}.device: type={dev.get('type')} anyOf={'anyOf' in dev} "
          f"x-widget={dev.get('x-widget')} x-device-kind={dev.get('x-device-kind')} default={dev.get('default')!r}")
    assert dev.get("type") == "string", f"{cls} 顶层 type 缺失"
    assert "anyOf" not in dev, f"{cls} 仍含 anyOf"
    assert dev.get("x-widget") == "audio-device-select"
    assert dev.get("x-device-kind") == kind
    assert dev.get("default") == ""

# ---- 2) 端点冒烟（mock 依赖）----
from fastapi.testclient import TestClient
from vrchat_agent.config import Config
from vrchat_agent.events import EventBus
from vrchat_agent.logging_setup import UsageTracker
from vrchat_agent.web import create_app

tmp = Path(tempfile.mkdtemp())
(tmp / "config.yaml").write_text(
    "web:\n  enabled: true\n  host: '127.0.0.1'\n  port: 0\n  auto_open_browser: false\n",
    encoding="utf-8",
)
cfg = Config.load(tmp / "config.yaml")
app = create_app(EventBus(), cfg, UsageTracker(), tmp)
c = TestClient(app)

r = c.get("/api/config/schema")
assert r.status_code == 200
defs = r.json()["schema"]["$defs"]
for cls, kind in [("AudioInputConfig", "input"), ("AudioOutputConfig", "output")]:
    dev = defs[cls]["properties"]["device"]
    print(f"schema端点 {cls}: type={dev.get('type')} anyOf={'anyOf' in dev} x-device-kind={dev.get('x-device-kind')}")
    assert dev.get("type") == "string" and "anyOf" not in dev

r2 = c.get("/api/audio/devices")
assert r2.status_code == 200
d = r2.json()
print("devices端点:", r2.status_code, "keys=", list(d.keys()), "input=", len(d["input"]), "output=", len(d["output"]))

# ---- 3) PUT 接受 device 空串（防回归契约）----
payload = cfg.model_dump()
payload["audio"]["input"]["device"] = ""
payload["audio"]["output"]["device"] = ""
payload["speech"]["output"]["device"] = ""
r3 = c.put("/api/config", json=payload)
print("PUT device空串:", r3.status_code, r3.json().get("ok"))
assert r3.status_code == 200 and r3.json()["ok"] is True

print("R2_SMOKE_OK")
