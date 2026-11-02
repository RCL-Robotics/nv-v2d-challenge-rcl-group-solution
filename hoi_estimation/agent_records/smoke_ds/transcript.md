# smoke_ds

冒烟测试  
模型: deepseek-flash api=api.deepseek.com:443 effort=low net=api  
日期: 2026-10-20T11:53:15Z

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/AGENT_TASK.md'
```
<details><summary>output</summary>

```
Write the text 'ok' into /work/outputs/hello.txt, run python to print 2+2, then stop.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/outputs && printf 'ok' > /work/outputs/hello.txt && python -c 'print(2+2)'"
```
<details><summary>output</summary>

```
4

```
</details>

**Agent:** Created `/work/outputs/hello.txt` with `ok`.

Python output: `4`
