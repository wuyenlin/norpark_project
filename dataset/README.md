The `get_img.sh` downloads the CCTV camera image from online and name it after the corresponding date and time in .jpg form. To run the task every hour, you need to download `crontab` and have it installed. 


## Setting up
### Step 1: Install package

```
sudo apt-get update && sudo apt-get upgrade
dpkg -l cron

sudo apt-get install cron
```

### Step 2: Ensure cron service is running

Run the following command:
```
systemctl status cron
```

### Step 3: Add job

To configure a new job, enter:
```
sudo vim /etc/crontab
```

Add a new line below, specifying the time and path to your file:
```
* * * * *   username ~/path/to/get_img.sh  # runs task every minute
@hourly     username ~/path/to/get_img.sh  # runs task hourly
```

The input format is as follows:
```
+-------+------------------------------------------+
| Field |                Description               |
+-------+------------------------------------------+
|   * 	| Minute                                   |
|   *   | Hour                                     |
|   *   | Every day                                |
|   *   | Every month                              |
|   *   | Every day of the week                    |
+-------+------------------------------------------+
```


### Step 4: Update changes
Finally, run
```
systemctl restart cron
```
to restart the crontab for the changes to take place. Insert your password when prompted. 
