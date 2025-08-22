package com.msdkremote;

import android.app.Application;
import android.content.Context;
import android.util.Log;

import dji.v5.common.error.IDJIError;
import dji.v5.common.register.DJISDKInitEvent;
import dji.v5.manager.SDKManager;
import dji.v5.manager.interfaces.SDKManagerCallback;

public class MyApplication extends Application
{
    private final String TAG = this.getClass().getSimpleName();

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(base);
        // Before calling install, do not call any MSDK related
        // Basically, chines viruses
        com.cySdkyc.clx.Helper.install(this);
    }
}
