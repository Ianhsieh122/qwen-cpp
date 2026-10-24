Remove-Item -Path telemetry_30s.csv -Force -ErrorAction SilentlyContinue
"Time,Power_W,GPU_Util_pct,Clock_MHz,MemClock_MHz,Temp_C,VRAM_MiB" | Out-File -FilePath telemetry_30s.csv -Encoding ascii

$job = Start-Job -ScriptBlock {
    while ($true) {
        $date = Get-Date -Format 'HH:mm:ss'
        $smi = nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.current.graphics,clocks.current.memory,temperature.gpu,memory.used --format=csv,noheader,nounits -i 0
        "$date,$smi" | Out-File -FilePath telemetry_30s.csv -Append -Encoding ascii
        Start-Sleep -Milliseconds 1000
    }
}

[System.Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', '0')
& 'build-win-cuda-v100\bin\llama-bench.exe' -m 'C:\Users\ianhs\OneDrive\30BB~1\lammacpp\models\Qwen3.8-27B-UD-Q3_K_XL.gguf' -ctk kvarn4 -ctv kvarn4 -ngl 99 -p 1024 -n 1024 -fa 1 | Out-File -FilePath bench_30s_result.txt -Encoding utf8

Stop-Job $job
Remove-Job $job
