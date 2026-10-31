package ai.sunnypilot.fold3;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import java.io.File;
import java.nio.file.Files;

public final class MainActivity extends Activity {
    private TextView status;
    private final Handler handler = new Handler();
    private boolean autoRun;
    private boolean soak;
    private boolean preparingUi;
    private boolean pendingVision;
    private UsbProbe usb;
    private final Runnable refresh = new Runnable() {
        public void run() {
            StringBuilder text = new StringBuilder();
            for (String name : new String[]{"progress.txt", "model-progress.txt", "usb-report.json", "panda-rx.json"}) {
                File report = new File(getFilesDir(), name);
                if (!report.isFile()) continue;
                try {
                    text.append(new String(Files.readAllBytes(report.toPath()), java.nio.charset.StandardCharsets.UTF_8)).append("\n\n");
                } catch (java.io.IOException e) { text.append("Cannot read ").append(name).append(": ").append(e.getMessage()).append('\n'); }
            }
            if (text.length() > 0) status.setText(text.toString());
            handler.postDelayed(this, 1000);
        }
    };
    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        usb = new UsbProbe(this);
        getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        autoRun = getIntent().getBooleanExtra("run_probe", false) && saved == null;
        soak = getIntent().getBooleanExtra("soak", false);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(24, 16, 24, 16);
        TextView title = new TextView(this);
        title.setText("sunnypilot · Fold 3 port validation\nIoniq PHEV 2019 · RHD · Factory ACC\nVehicle transmission disabled: port qualification incomplete");
        title.setTextSize(18);
        layout.addView(title);
        Button probe = new Button(this);
        probe.setText("Test built-in camera combinations");
        probe.setOnClickListener(v -> start(false));
        layout.addView(probe);
        if (BuildConfig.DEBUG && getPackageName().equals("com.samsung.sunnypilot.fold3")) {
            Button multiview = new Button(this);
            multiview.setText("Test Samsung multiview + front (diagnostic)");
            multiview.setOnClickListener(v -> {
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
                stopService(new Intent(this, VisionService.class));
                startForegroundService(new Intent(this, ProbeService.class).putExtra("samsung_multiview", true)
                    .putExtra("front_camera", true).putExtra("duration_seconds", 10));
            });
            layout.addView(multiview);
        }
        Button soakButton = new Button(this);
        soakButton.setText("Run 30-minute road + driver camera test");
        soakButton.setOnClickListener(v -> start(true));
        layout.addView(soakButton);
        Button models = new Button(this);
        models.setText("Validate original models on OpenCL");
        models.setOnClickListener(v -> startForegroundService(new Intent(this, ModelService.class)));
        layout.addView(models);
        Button dependencies = new Button(this);
        dependencies.setText("Inspect runtime dependencies");
        dependencies.setOnClickListener(v -> startService(new Intent(this, RuntimeService.class)));
        layout.addView(dependencies);
        Button stop = new Button(this);
        stop.setText("Stop camera test");
        stop.setOnClickListener(v -> stopService(new Intent(this, ProbeService.class)));
        layout.addView(stop);
        Button panda = new Button(this);
        panda.setText("Read Panda USB identity (no transmission)");
        panda.setOnClickListener(v -> usb.start());
        layout.addView(panda);
        Button pandaRx = new Button(this);
        pandaRx.setText("Start silent Panda receive-only diagnostics");
        pandaRx.setOnClickListener(v -> usb.start(true));
        layout.addView(pandaRx);
        Button stopPanda = new Button(this);
        stopPanda.setText("Stop Panda receive-only diagnostics");
        stopPanda.setOnClickListener(v -> stopService(new Intent(this, PandaRxService.class)));
        layout.addView(stopPanda);
        Button vision = new Button(this);
        vision.setText("Start live road + driver visual diagnostics");
        vision.setOnClickListener(v -> startVision(getIntent()));
        layout.addView(vision);
        Button stopVision = new Button(this);
        stopVision.setText("Stop live cameras");
        stopVision.setOnClickListener(v -> stopService(new Intent(this, VisionService.class)));
        layout.addView(stopVision);
        status = new TextView(this);
        status.setText("Ready. Reports are stored locally. Camera testing closes each session before opening the next.");
        ScrollView scroll = new ScrollView(this);
        scroll.addView(status);
        layout.addView(scroll);
        setContentView(layout);
        Button runtime = new Button(this);
        runtime.setText("Start supervised runtime (transmission disabled)");
        runtime.setOnClickListener(v -> startForegroundService(new Intent(this, ManagerService.class)));
        layout.addView(runtime, 0);
        Button drivingUi = new Button(this);
        drivingUi.setText("Open Raylib UI");
        drivingUi.setOnClickListener(v -> launchDrivingUi());
        layout.addView(drivingUi, 1);
        Button stopRuntime = new Button(this);
        stopRuntime.setText("Stop supervised runtime");
        stopRuntime.setOnClickListener(v -> stopService(new Intent(this, ManagerService.class)));
        layout.addView(stopRuntime, 2);
        if (saved == null && getIntent().getBooleanExtra("run_runtime", false))
            startForegroundService(new Intent(this, ManagerService.class).putExtra("simulation", getIntent().getBooleanExtra("simulation", false)));
        if (saved == null && getIntent().getBooleanExtra("raylib", false)) launchDrivingUi();
        if (saved == null && getIntent().getBooleanExtra("run_vision", false)) startVision(getIntent());
        if (getIntent().getBooleanExtra("stop_vision", false)) stopService(new Intent(this, VisionService.class));
        if (saved == null && BuildConfig.DEBUG && getIntent().getBooleanExtra("panda_rx", false)) usb.start(true);
        if (!pendingVision && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.POST_NOTIFICATIONS}, 1);
        } else if (autoRun) { autoRun = false; start(soak); }
        if (saved == null && getIntent().getBooleanExtra("inspect_runtime", false))
            startService(new Intent(this, RuntimeService.class));
        if (saved == null && getIntent().getBooleanExtra("run_models", false)) startForegroundService(new Intent(this, ModelService.class)
            .putExtra("model_profile", getIntent().getStringExtra("model_profile")));
    }
    private void startVision(Intent intent) {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            pendingVision = true;
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 2);
            return;
        }
        stopService(new Intent(this, ProbeService.class));
        stopService(new Intent(this, ManagerService.class));
        startForegroundService(new Intent(this, ManagerService.class));
        startActivity(new Intent(this, DrivingActivity.class).putExtra("preview", true)
            .putExtra("start_vision", true).putExtra("vision_seconds", intent.getIntExtra("vision_seconds", 1800))
            .putExtra("samsung_multiview", BuildConfig.DEBUG && intent.getBooleanExtra("samsung_multiview", false)));
    }
    private void launchDrivingUi() {
        if (preparingUi) return;
        preparingUi = true;
        new Thread(() -> {
            try {
                PythonRuntime.start(this);
                runOnUiThread(() -> { preparingUi = false; if (!isFinishing()) startActivity(new Intent(this, DrivingActivity.class)
                    .addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
                    .putExtra("simulation", getIntent().getBooleanExtra("simulation", false))); });
            } catch (Exception e) { runOnUiThread(() -> { preparingUi = false; status.setText("UI startup failed: " + e); }); }
        }, "prepare-ui").start();
    }
    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        if (intent.getBooleanExtra("run_runtime", false))
            startForegroundService(new Intent(this, ManagerService.class).putExtra("simulation", intent.getBooleanExtra("simulation", false)));
        if (intent.getBooleanExtra("raylib", false)) launchDrivingUi();
        if (intent.getBooleanExtra("run_vision", false)) startVision(intent);
        if (intent.getBooleanExtra("stop_vision", false)) stopService(new Intent(this, VisionService.class));
        if (BuildConfig.DEBUG && intent.getBooleanExtra("panda_rx", false)) usb.start(true);
        if (intent.getBooleanExtra("stop_panda", false)) stopService(new Intent(this, PandaRxService.class));
    }
    private void start(boolean runSoak) {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
        stopService(new Intent(this, VisionService.class));
        startForegroundService(new Intent(this, ProbeService.class)
            .putExtras(BuildConfig.DEBUG ? getIntent() : new Intent()).putExtra("soak", runSoak)
            .putExtra("soak_only", getIntent().getBooleanExtra("soak_only", false)));
    }
    @Override public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (pendingVision && checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            pendingVision = false; startVision(getIntent());
        }
        if (autoRun && checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            autoRun = false; start(soak);
        }
    }
    @Override public void onResume() { super.onResume(); handler.post(refresh); }
    @Override public void onPause() { handler.removeCallbacks(refresh); super.onPause(); }
    @Override public void onDestroy() { usb.close(); super.onDestroy(); }
}
