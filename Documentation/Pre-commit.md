## **Setting Up Pre-commit**

**Pre-commit** is a framework for managing and maintaining multi-language pre-commit hooks to ensure code quality and consistency. It helps identify simple issues before submission to code review and maintain code standards. We have set up pre-commit hooks for linting and auto-formatting code to ensure code quality and consistency.

Please refer to [pre-commit documentation](https://pre-commit.com/) for more detailed information, including advanced configuration and usage.

## **Prerequisites**

Before setting up pre-commit hooks in your development environment, ensure you meet the following prerequisites:

[**Python**](https://www.python.org/)

Pre-commit is a Python-based tool. If you're unsure which version you have, you can check by running:

```bash
python --version
```
or
```bash
python3 --version
```

 [**Git**](https://git-scm.com/)

Pre-commit hooks work within the Git version control system. To verify if Git is installed, run:

```bash
git --version
```

If Git is not installed, you can download it from the [official Git website](https://git-scm.com/downloads). Follow the installation instructions for your operating system. After installation, configure Git with your user name and email address using the following commands:

```bash
git config --global user.name "Your Name"
git config --global user.email "youremail@example.com"
```

## **Installation**

1. **Install pre-commit:**
   
   You can install pre-commit globally using pip:
   ```bash
   pip install pre-commit
   ```

2. **Clone the Repository:**

   If you haven't already, clone the repository to your local device:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. **Install the Git Hook Scripts:**

   Run the following command in the repository's root directory. This will install the pre-commit script in your `.git/hooks` directory.

   ```bash
   pre-commit install
   ```

   Optionally, if you want to set up pre-commit hooks for the **commit-msg** hook, run:

   ```bash
   pre-commit install --hook-type commit-msg
   ```

4. **Running Pre-commit:**

   After installation, pre-commit will run automatically on `git commit` for the files you've staged. 

   ```bash
   pre-commit run --all-files
   ```

5. **Updating Hooks:**

   To update the hooks to the latest versions specified in the `.pre-commit-config.yaml` file, run:

   ```bash
   pre-commit autoupdate
   ```


## Ensuring `commitlint` Runs on Commit Messages

The `commitlint` hook is configured to run at the `commit-msg` stage. This means it will only execute when processing the commit message, not during the pre-commit checks of the files. Here's how to ensure it runs as expected.

### Step 1: Install `commitlint` for the Commit Message Stage

To enable `commitlint` for commit messages, you need to install the pre-commit hook specifically for the `commit-msg` stage:

```bash
pre-commit install --hook-type commit-msg
```

This command sets up the pre-commit framework to run hooks at the `commit-msg` stage.

### Step 2: Verify Hook Installation

After installation, ensure that the hooks for both the pre-commit and commit-msg stages are correctly installed:

```bash
pre-commit install -t pre-commit
pre-commit install -t commit-msg
```

This ensures that hooks for checking your files (`pre-commit`) and commit messages (`commit-msg`) are both set up.

### Step 3: Manual Triggering (Optional)

If you wish to manually trigger the `commit-msg` hooks to test their functionality, you can use the following command:

```bash
echo "Your commit message" | pre-commit run --hook-stage commit-msg --all-files
```

## **Troubleshooting**

- **Skipping Hooks Temporarily** 

    If you need to commit changes but bypass the pre-commit checks temporarily, you can use the `--no-verify` option with `git commit`:

  ```bash
  git commit --no-verify -m "Your commit message"
  ```

- **Resolving Hook Failures** 

    If a hook fails, follow the instructions provided in the output to resolve the issue. You may need to run 
  ```bash
  pre-commit run --all-files
   ```
     to test all files, make necessary changes, and then try committing again. You can run the following code to generate a log file in your current folder to read the failure result.
   ```bash
   pre-commit run --all-files | tee pre-commit-log.txt 
  ```
