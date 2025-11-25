# Story Architect - Streamlit Cloud Deployment Guide

Complete guide for deploying Story Architect to Streamlit Community Cloud with private access.

---

## 🚀 Quick Deployment Checklist

- [ ] GitHub repository is ready (public or private)
- [ ] All secrets documented and ready to paste
- [ ] `.gitignore` properly configured (no API keys committed)
- [ ] Test app works locally
- [ ] Streamlit Community Cloud account created
- [ ] Repository connected to Streamlit Cloud
- [ ] Secrets configured in Cloud dashboard
- [ ] Viewers added for private access

---

## Prerequisites

### 1. GitHub Repository

Your code must be in a GitHub repository (public or private):

```bash
# If not already initialized
git init
git add .
git commit -m "Initial commit"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/story-architect.git
git push -u origin master
```

**Important:** Ensure `.gitignore` excludes secrets:
- ✓ `.env` (local secrets)
- ✓ `.streamlit/secrets.toml` (local Streamlit secrets)
- ✓ `google-credentials.json` (service account keys)

### 2. API Keys Ready

Have these API keys ready to configure:

| API | Where to Get | Required? |
|-----|--------------|-----------|
| Google Gemini | [ai.google.dev](https://aistudio.google.com/apikey) | ✅ Required |
| FAL.ai | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) | ✅ Required |
| Google Cloud Project | [console.cloud.google.com](https://console.cloud.google.com) | ⚠️ Optional (Vertex AI) |

---

## Step 1: Test Locally

Before deploying, verify everything works locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Create secrets file (if not already created)
# .streamlit/secrets.toml should contain your API keys

# Run the app
streamlit run gui/app.py
```

Visit `http://localhost:8501` and test:
1. ✓ API keys detected in sidebar
2. ✓ Create a test project
3. ✓ Verify no errors in console

---

## Step 2: Create Streamlit Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Authorize Streamlit to access your repositories

---

## Step 3: Deploy Your App

### 3.1 Create New App

1. Click **"New app"** button
2. Choose your repository: `YOUR_USERNAME/story-architect`
3. Select branch: `master` (or `main`)
4. Set main file path: `gui/app.py`
5. (Optional) Choose a custom subdomain

### 3.2 Advanced Settings

**App URL:** `your-custom-name.streamlit.app`

**Python version:** 3.10+ (recommended)

Click **"Deploy!"**

---

## Step 4: Configure Secrets (Critical!)

After deployment starts, immediately configure secrets:

### 4.1 Access Secrets Panel

1. Go to your app dashboard
2. Click **"Settings"** (⚙️ icon)
3. Select **"Secrets"** tab

### 4.2 Add Your Secrets

Copy and paste the following template, replacing with your actual values:

```toml
# Google Gemini API Keys
GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_KEY_HERE"
GOOGLE_API_KEY = "YOUR_ACTUAL_GEMINI_KEY_HERE"

# FAL.ai API Key
FAL_KEY = "YOUR_ACTUAL_FAL_KEY_HERE"

# Google Cloud / Vertex AI (Optional)
GOOGLE_CLOUD_PROJECT = "your-project-id"
GOOGLE_CLOUD_LOCATION = "global"
GOOGLE_APPLICATION_CREDENTIALS = "google-credentials.json"

# Optional Settings
LOG_LEVEL = "INFO"
MAX_RETRIES = "3"
```

**Important Notes:**
- Do NOT include comments in the actual secrets editor
- Use the exact key names shown above
- Keys are case-sensitive
- No quotes needed around values in secrets.toml format

### 4.3 Save and Reboot

1. Click **"Save"**
2. App will automatically reboot with new secrets
3. Wait 1-2 minutes for reboot to complete

---

## Step 5: Configure Private Access

Since your GitHub repo is private, your app will be **private by default**. Here's how to control access:

### 5.1 Access Control Basics

**Private App (default):**
- Only visible to developers in your workspace
- Must add viewers manually
- Viewers need email-based authentication

**Public App (if repo is public):**
- Anyone with the URL can access
- No authentication required
- Not recommended for apps with usage costs

### 5.2 Add Viewers

To allow specific people to access your private app:

1. Go to app **Settings** → **Sharing**
2. Under **"Viewers"**, click **"Invite viewers"**
3. Add email addresses (one per line):
   ```
   colleague1@example.com
   colleague2@example.com
   boss@example.com
   ```
4. Click **"Save"**

### 5.3 Viewer Authentication

When a viewer accesses your app:
- **Google account exists:** Sign in with Google OAuth
- **No Google account:** Receive single-use email link
- Authentication required for every session

### 5.4 Managing Access

**To remove a viewer:**
1. Settings → Sharing → Viewers
2. Click **"Remove"** next to their email
3. They'll lose access immediately

**To add developers:**
1. Add them as collaborators on GitHub repo
2. They'll automatically have developer access
3. Developers can edit settings and view logs

---

## Step 6: Verify Deployment

### 6.1 Check App Health

1. Visit your app URL: `https://your-app.streamlit.app`
2. Verify you can sign in (if private)
3. Check sidebar shows **"🔐 Using Streamlit Secrets"**
4. Verify API keys show as **"✓ Configured"**

### 6.2 Test Functionality

Run a complete test:
1. Go to **"New Project"** page
2. Enter a short story concept
3. Click **"Run Pipeline"**
4. Monitor for any errors
5. Verify agents complete successfully

### 6.3 Review Logs (If Errors Occur)

If something breaks:
1. App dashboard → **"Manage app"**
2. Click **"Logs"** tab
3. Review error messages
4. Check for missing secrets or permissions

---

## Step 7: Monitor Usage & Costs

### 7.1 Streamlit Community Cloud Limits

**Free tier includes:**
- ✓ Unlimited public apps
- ✓ 1 private app
- ✓ Up to 1 GB storage per app
- ✓ Shared compute resources

**Upgrade for:**
- More private apps
- Custom authentication
- Priority support
- More compute resources

### 7.2 API Usage Costs

**Your costs come from API usage, not Streamlit:**

| API | Free Tier | Paid Tier | Cost |
|-----|-----------|-----------|------|
| Google Gemini | 15 RPM | 1000+ RPM | $0.00 - $0.075/1K tokens |
| FAL.ai | Limited | Unlimited | $0.003 - $0.05 per image |
| Vertex AI Veo | N/A | Pay-per-use | ~$0.10 per video |

**Recommendation:** Start with free tiers, upgrade as needed.

### 7.3 Monitor API Usage

**Google Gemini:**
- Dashboard: [ai.google.dev](https://aistudio.google.com)
- Enable billing for higher limits

**FAL.ai:**
- Dashboard: [fal.ai/dashboard](https://fal.ai/dashboard)
- Usage → Billing → View costs

---

## Troubleshooting

### Problem: "API Key Not Found"

**Solution:**
1. Check secrets are saved in Streamlit Cloud (Settings → Secrets)
2. Verify exact key names: `GEMINI_API_KEY`, `FAL_KEY`
3. Reboot app after saving secrets
4. Check logs for specific error messages

### Problem: "Module not found"

**Solution:**
1. Ensure `requirements.txt` includes all dependencies
2. Check Python version compatibility (3.10+ recommended)
3. Reboot app or redeploy

### Problem: "Permission denied" for Google Cloud

**Solution:**
1. Upload `google-credentials.json` to GitHub (encrypted secret)
2. OR paste JSON content directly in secrets:
   ```toml
   GOOGLE_APPLICATION_CREDENTIALS_JSON = '''
   {
     "type": "service_account",
     "project_id": "your-project",
     ...
   }
   '''
   ```
3. Update code to write JSON to file at runtime

### Problem: App is slow or times out

**Solution:**
1. Streamlit Cloud has 1 GB memory limit
2. Large video generation may exceed limits
3. Consider:
   - Reducing video durations
   - Processing in batches
   - Using Vertex AI (better quotas)

### Problem: Viewers can't access app

**Solution:**
1. Ensure repo is set to private (Settings → General → Visibility)
2. Add viewer emails in Streamlit (Settings → Sharing)
3. Viewers must sign in with that email
4. Check spam folder for email verification links

---

## Updating Your App

### Auto-Deploy

Streamlit Cloud automatically deploys on git push:

```bash
# Make changes to code
git add .
git commit -m "Update feature X"
git push

# App will redeploy automatically (1-2 minutes)
```

### Manual Reboot

To restart without code changes:
1. App dashboard → **"Manage app"**
2. Click **"Reboot app"**
3. Wait 1-2 minutes

### Rollback to Previous Version

1. Revert Git commit:
   ```bash
   git revert HEAD
   git push
   ```
2. Or use GitHub UI to revert commit
3. App auto-redeploys previous version

---

## Security Best Practices

### ✅ DO:
- Use Streamlit Secrets for all API keys
- Add `.streamlit/secrets.toml` to `.gitignore`
- Never commit `.env` or credentials files
- Rotate API keys regularly
- Use Vertex AI service accounts for production
- Set app to private if handling sensitive data
- Monitor API usage and costs

### ❌ DON'T:
- Hardcode API keys in source code
- Commit secrets to Git (even private repos)
- Share your Streamlit app URL publicly (if private)
- Use production keys for testing
- Leave unused viewer access active

---

## Advanced: Custom Domain

To use your own domain (e.g., `story.yourdomain.com`):

1. Upgrade to Streamlit Teams ($250+/month)
2. OR deploy on your own infrastructure:
   - AWS/GCP with Docker
   - Heroku
   - DigitalOcean
   - Self-hosted with NGINX

---

## Support & Resources

### Streamlit Community Cloud
- Docs: [docs.streamlit.io](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- Forum: [discuss.streamlit.io](https://discuss.streamlit.io)
- Status: [status.streamlit.io](https://status.streamlit.io)

### Story Architect
- Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/story-architect/issues)
- Local docs: See `README.md` in project root

---

## Next Steps

After successful deployment:

1. ✓ Share URL with viewers
2. ✓ Set up monitoring for API costs
3. ✓ Create backups of generated content
4. ✓ Configure automated exports
5. ✓ Document custom workflows for your team

**Your app is now live! 🎉**

Visit: `https://your-app.streamlit.app`
