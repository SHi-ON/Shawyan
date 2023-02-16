---
Title: Mimic GitHub PR Diff Locally
Date: 2022-01-02 21:42
Category: Development
---

## What?!
The goal is to mimic the GitHub pull requests' file diff in your local repository:
![Image]({attach}post_images/mimic_github_pr_diff_locally_1.png)


## How?
```shell
$ git diff <FIRTS_BRANCH_COMMIT_SHA> <LAST_BRANCH_COMMIT_SHA> --stat
```
To get a cleaner summary, pipe it to the `tail` command:
```shell
$ git diff <FIRTS_BRANCH_COMMIT_SHA> <LAST_BRANCH_COMMIT_SHA> --stat | tail -1

```
and you will get:
![Image]({attach}post_images/mimic_github_pr_diff_locally_2.png)