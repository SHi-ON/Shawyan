---
title: "Mimic GitHub PR Diff Locally"
date: 2022-01-02T21:42:42-04:00
draft: false
author: SHi-ON
---

## What?!
The goal is to mimic the GitHub pull requests' change diff in your local repository:
![Image](../post_images/mimic_github_pr_diff_locally_1.png)


## How?
```shell
$ git diff <FIRTS_BRANCH_COMMIT_SHA> <LAST_BRANCH_COMMIT_SHA> --stat
```
To get a cleaner summary, pipe it to the `tail` command:
```shell
$ git diff <FIRTS_BRANCH_COMMIT_SHA> <LAST_BRANCH_COMMIT_SHA> --stat | tail -1

```
and you will get:
![Image](../post_images/mimic_github_pr_diff_locally_2.png)