# 🐳 Docker Build & Push Troubleshooting Guide

## 🚨 Current Issue: 403 Forbidden Error

The CI/CD pipeline is failing with:
```
ERROR: failed to push ghcr.io/mjahmadee/cataract-lmm:main: unexpected status from HEAD request to https://ghcr.io/v2/mjahmadee/cataract-lmm/blobs/sha256:...: 403 Forbidden
```

## 🔍 Root Cause Analysis

This error occurs due to insufficient permissions to push Docker images to GitHub Container Registry (GHCR). The main causes are:

1. **Repository Permissions**: The default `GITHUB_TOKEN` doesn't have `packages:write` scope
2. **Workflow Permissions**: GitHub Actions workflow permissions are too restrictive
3. **Package Visibility**: Container registry package settings may be misconfigured

## 🛠️ Solutions (Choose One)

### ✅ Solution 1: Fix Repository Workflow Permissions (Recommended)

**Step 1**: Go to your repository settings
1. Navigate to `https://github.com/MJAHMADEE/Cataract-LMM/settings/actions`
2. Under **"Workflow permissions"**, select:
   - ✅ **"Read and write permissions"**
   - ✅ **"Allow GitHub Actions to create and approve pull requests"**
3. Click **"Save"**

**Step 2**: Verify package settings
1. Go to `https://github.com/MJAHMADEE/Cataract-LMM/settings/packages`
2. Ensure the package `cataract-lmm` exists and has proper permissions
3. Set visibility to **Public** or ensure your token has access

### 🔐 Solution 2: Use Personal Access Token (Alternative)

**Step 1**: Create a Personal Access Token
1. Go to `https://github.com/settings/tokens`
2. Click **"Generate new token (classic)"**
3. Select scopes:
   - ✅ `write:packages`
   - ✅ `read:packages` 
   - ✅ `delete:packages` (optional)
4. Copy the token

**Step 2**: Add token as repository secret
1. Go to `https://github.com/MJAHMADEE/Cataract-LMM/settings/secrets/actions`
2. Click **"New repository secret"**
3. Name: `GHCR_TOKEN`
4. Value: Your PAT from Step 1

**Step 3**: Update workflow to use PAT
```yaml
- name: 🔑 Log in to Container Registry
  if: github.event_name != 'pull_request'
  uses: docker/login-action@v3
  with:
    registry: ${{ env.REGISTRY }}
    username: ${{ github.actor }}
    password: ${{ secrets.GHCR_TOKEN }}  # Changed from GITHUB_TOKEN
```

### 🔧 Solution 3: Use Docker Hub Instead (Fallback)

**Step 1**: Create Docker Hub account and repository
1. Sign up at https://hub.docker.com/
2. Create repository: `mjahmadee/cataract-lmm`

**Step 2**: Add Docker Hub secrets
1. Go to repository settings → Secrets and variables → Actions
2. Add secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token

**Step 3**: Update workflow environment variables
```yaml
env:
  REGISTRY: docker.io
  IMAGE_NAME: mjahmadee/cataract-lmm
```

## 🚀 Testing the Fix

After implementing Solution 1 (recommended):

1. **Push a small change** to trigger the workflow:
   ```bash
   git commit --allow-empty -m "test: trigger workflow for docker fix"
   git push
   ```

2. **Monitor the workflow** at:
   `https://github.com/MJAHMADEE/Cataract-LMM/actions`

3. **Expected result**: Docker build and push should succeed

## 📋 Current Workflow Status

The workflow has been updated to:
- ✅ Continue on Docker push failure (non-blocking)
- ✅ Build local Docker image if GHCR push fails
- ✅ Provide helpful error messages and solutions
- ✅ Run security scans on locally built images

## 🔍 Verification Commands

After the fix, you should be able to pull your image:
```bash
# Once permissions are fixed
docker pull ghcr.io/mjahmadee/cataract-lmm:main

# Verify the image
docker run --rm ghcr.io/mjahmadee/cataract-lmm:main python --version
```

## 📊 Additional Notes

- **Build Time**: Multi-architecture builds (amd64, arm64) take ~15-30 minutes
- **Image Size**: Optimized with multi-stage builds and dependency caching
- **Security**: Images are scanned with Trivy for vulnerabilities
- **Caching**: GitHub Actions cache is used to speed up subsequent builds

## 🆘 Still Having Issues?

If you continue experiencing problems:

1. **Check workflow logs** for detailed error messages
2. **Verify repository permissions** are correctly set
3. **Test with a simple push** to a test repository first
4. **Contact GitHub Support** for persistent GHCR permission issues

---

**Last Updated**: September 9, 2025
**Status**: ✅ Workflow updated with fallback handling
**Next Action**: Fix repository workflow permissions (Solution 1)
