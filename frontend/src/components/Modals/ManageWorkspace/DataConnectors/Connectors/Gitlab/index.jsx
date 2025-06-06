import React, { useEffect, useState } from "react";
import System from "@/models/system";
import showToast from "@/utils/toast";
import pluralize from "pluralize";
import { TagsInput } from "react-tag-input-component";
import { Info, Warning } from "@phosphor-icons/react";
import { Tooltip } from "react-tooltip";
import { useTranslation } from "react-i18next";

const DEFAULT_BRANCHES = ["main", "master"];
export default function GitlabOptions() {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [repo, setRepo] = useState(null);
  const [accessToken, setAccessToken] = useState(null);
  const [ignores, setIgnores] = useState([]);
  const [settings, setSettings] = useState({
    repo: null,
    accessToken: null,
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    const form = new FormData(e.target);

    try {
      setLoading(true);
      showToast(
        "Fetching all files for repo - this may take a while.",
        "info",
        { clear: true, autoClose: false }
      );
      const { data, error } = await System.dataConnectors.gitlab.collect({
        repo: form.get("repo"),
        accessToken: form.get("accessToken"),
        branch: form.get("branch"),
        ignorePaths: ignores,
        fetchIssues: form.get("fetchIssues"),
        fetchWikis: form.get("fetchWikis"),
      });

      if (!!error) {
        showToast(error, "error", { clear: true });
        setLoading(false);
        return;
      }

      showToast(
        `${data.files} ${pluralize("file", data.files)} collected from ${
          data.author
        }/${data.repo}:${data.branch}. Output folder is ${data.destination}.`,
        "success",
        { clear: true }
      );
      e.target.reset();
      setLoading(false);
      return;
    } catch (e) {
      console.error(e);
      showToast(e.message, "error", { clear: true });
      setLoading(false);
    }
  };

  return (
    <div className="flex w-full">
      <div className="flex flex-col w-full px-1 md:pb-6 pb-16">
        <form className="w-full" onSubmit={handleSubmit}>
          <div className="w-full flex flex-col py-2">
            <div className="w-full flex flex-col gap-4">
              <div className="flex flex-col pr-10">
                <div className="flex flex-col gap-y-1 mb-4">
                  <label className="text-white text-sm font-bold">
                    {t("connectors.gitlab.URL")}
                  </label>
                  <p className="text-xs font-normal text-theme-text-secondary">
                    {t("connectors.gitlab.URL_explained")}
                  </p>
                </div>
                <input
                  type="url"
                  name="repo"
                  className="border-none bg-theme-settings-input-bg text-white placeholder:text-theme-settings-input-placeholder text-sm rounded-lg focus:outline-primary-button active:outline-primary-button outline-none block w-full p-2.5"
                  placeholder="https://gitlab.com/gitlab-org/gitlab"
                  required={true}
                  autoComplete="off"
                  onChange={(e) => setRepo(e.target.value)}
                  onBlur={() => setSettings({ ...settings, repo })}
                  spellCheck={false}
                />
              </div>
              <div className="flex flex-col pr-10">
                <div className="flex flex-col gap-y-1 mb-4">
                  <label className="text-white font-bold text-sm flex gap-x-2 items-center">
                    <p className="font-bold text-white">
                      {t("connectors.gitlab.token")}
                    </p>{" "}
                    <p className="text-xs font-light flex items-center">
                      <span className="text-theme-text-secondary">
                        {t("connectors.gitlab.optional")}
                      </span>
                      <PATTooltip accessToken={accessToken} />
                    </p>
                  </label>
                  <p className="text-xs font-normal text-theme-text-secondary">
                    {t("connectors.gitlab.token_description")}
                  </p>
                </div>
                <input
                  type="text"
                  name="accessToken"
                  className="border-none bg-theme-settings-input-bg text-white placeholder:text-theme-settings-input-placeholder text-sm rounded-lg focus:outline-primary-button active:outline-primary-button outline-none block w-full p-2.5"
                  placeholder="glpat-XXXXXXXXXXXXXXXXXXXX"
                  required={false}
                  autoComplete="off"
                  spellCheck={false}
                  onChange={(e) => setAccessToken(e.target.value)}
                  onBlur={() => setSettings({ ...settings, accessToken })}
                />
              </div>
              <div className="flex flex-col pr-10">
                <div className="flex flex-col gap-y-1 mb-4">
                  <label className="text-white font-bold text-sm flex gap-x-2 items-center">
                    <p className="font-bold text-white">Settings</p>
                  </label>
                  <p className="text-xs font-normal text-white">
                    {t("connectors.gitlab.token_description")}
                  </p>
                </div>
                <div className="flex items-center gap-x-2 mb-3">
                  <label className="relative inline-flex cursor-pointer items-center">
                    <input
                      type="checkbox"
                      name="fetchIssues"
                      value={true}
                      className="peer sr-only"
                    />
                    <div className="peer-disabled:opacity-50 pointer-events-none peer h-6 w-11 rounded-full bg-[#CFCFD0] after:absolute after:left-[2px] after:top-[2px] after:h-5 after:w-5 after:rounded-full after:shadow-xl after:border-none after:bg-white after:box-shadow-md after:transition-all after:content-[''] peer-checked:bg-[#32D583] peer-checked:after:translate-x-full peer-checked:after:border-white peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-transparent"></div>
                    <span className="ml-3 text-sm font-medium text-white">
                      {t("connectors.gitlab.fetch_issues")}
                    </span>
                  </label>
                </div>
                <div className="flex items-center gap-x-2">
                  <label className="relative inline-flex cursor-pointer items-center">
                    <input
                      type="checkbox"
                      name="fetchWikis"
                      value={true}
                      className="peer sr-only"
                    />
                    <div className="peer-disabled:opacity-50 pointer-events-none peer h-6 w-11 rounded-full bg-[#CFCFD0] after:absolute after:left-[2px] after:top-[2px] after:h-5 after:w-5 after:rounded-full after:shadow-xl after:border-none after:bg-white after:box-shadow-md after:transition-all after:content-[''] peer-checked:bg-[#32D583] peer-checked:after:translate-x-full peer-checked:after:border-white peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-transparent"></div>
                    <span className="ml-3 text-sm font-medium text-white">
                      Fetch Wikis as Documents
                    </span>
                  </label>
                </div>
              </div>
              <GitLabBranchSelection
                repo={settings.repo}
                accessToken={settings.accessToken}
              />
            </div>

            <div className="flex flex-col w-full py-4 pr-10">
              <div className="flex flex-col gap-y-1 mb-4">
                <label className="text-white text-sm flex gap-x-2 items-center">
                  <p className="text-white text-sm font-bold">
                    {t("connectors.gitlab.ignores")}
                  </p>
                </label>
                <p className="text-xs font-normal text-theme-text-secondary">
                  {t("connectors.gitlab.git_ignore")}
                </p>
              </div>
              <TagsInput
                value={ignores}
                onChange={setIgnores}
                name="ignores"
                placeholder="!*.js, images/*, .DS_Store, bin/*"
                classNames={{
                  tag: "bg-theme-settings-input-bg light:bg-black/10 bg-blue-300/10 text-zinc-800",
                  input:
                    "flex p-1 !bg-theme-settings-input-bg text-white placeholder:text-theme-settings-input-placeholder text-sm rounded-lg focus:outline-primary-button active:outline-primary-button outline-none",
                }}
              />
            </div>
          </div>

          <div className="flex flex-col gap-y-2 w-full pr-10">
            <PATAlert accessToken={accessToken} />
            <button
              type="submit"
              disabled={loading}
              className="mt-2 w-full justify-center border-none px-4 py-2 rounded-lg text-dark-text light:text-white text-sm font-bold items-center flex gap-x-2 bg-theme-home-button-primary hover:bg-theme-home-button-primary-hover disabled:bg-theme-home-button-primary-hover disabled:cursor-not-allowed"
            >
              {loading ? "Collecting files..." : "Submit"}
            </button>
            {loading && (
              <p className="text-xs text-white/50">
                {t("connectors.gitlab.task_explained")}
              </p>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

function GitLabBranchSelection({ repo, accessToken }) {
  const { t } = useTranslation();
  const [allBranches, setAllBranches] = useState(DEFAULT_BRANCHES);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAllBranches() {
      if (!repo) {
        setAllBranches(DEFAULT_BRANCHES);
        setLoading(false);
        return;
      }

      setLoading(true);
      const { branches } = await System.dataConnectors.gitlab.branches({
        repo,
        accessToken,
      });
      setAllBranches(branches.length > 0 ? branches : DEFAULT_BRANCHES);
      setLoading(false);
    }
    fetchAllBranches();
  }, [repo, accessToken]);

  if (loading) {
    return (
      <div className="flex flex-col w-60">
        <div className="flex flex-col gap-y-1 mb-4">
          <label className="text-white text-sm font-bold">
            {t("connectors.gitlab.branch")}
          </label>
          <p className="text-xs font-normal text-theme-text-secondary">
            {t("connectors.gitlab.branch_explained")}
          </p>
        </div>
        <select
          name="branch"
          required={true}
          className="border-none bg-theme-settings-input-bg border-gray-500 text-white focus:outline-primary-button active:outline-primary-button outline-none text-sm rounded-lg block w-full p-2.5"
        >
          <option disabled={true} selected={true}>
            {t("connectors.gitlab.branch_loading")}
          </option>
        </select>
      </div>
    );
  }

  return (
    <div className="flex flex-col w-60">
      <div className="flex flex-col gap-y-1 mb-4">
        <label className="text-white text-sm font-bold">Branch</label>
        <p className="text-xs font-normal text-theme-text-secondary">
          {t("connectors.gitlab.branch_explained")}
        </p>
      </div>
      <select
        name="branch"
        required={true}
        className="border-none bg-theme-settings-input-bg border-gray-500 text-white focus:outline-primary-button active:outline-primary-button outline-none text-sm rounded-lg block w-full p-2.5"
      >
        {allBranches.map((branch) => {
          return (
            <option key={branch} value={branch}>
              {branch}
            </option>
          );
        })}
      </select>
    </div>
  );
}

function PATAlert({ accessToken }) {
  const { t } = useTranslation();
  if (!!accessToken) return null;
  return (
    <div className="flex flex-col md:flex-row md:items-center gap-x-2 text-white mb-4 bg-blue-800/30 w-fit rounded-lg px-4 py-2">
      <div className="gap-x-2 flex items-center">
        <Info className="shrink-0" size={25} />
        <p className="text-sm">
          <span
            dangerouslySetInnerHTML={{
              __html: t("connectors.gitlab.token_information"),
            }}
          />
          <br />
          <br />
          <a
            href="https://gitlab.com/-/user_settings/personal_access_tokens"
            rel="noreferrer"
            target="_blank"
            className="underline"
            onClick={(e) => e.stopPropagation()}
          >
            {t("connectors.gitlab.token_personal")}
          </a>
        </p>
      </div>
    </div>
  );
}

function PATTooltip({ accessToken }) {
  const { t } = useTranslation();
  if (!!accessToken) return null;
  return (
    <>
      {!accessToken && (
        <Warning
          size={14}
          className="ml-1 text-orange-500 cursor-pointer"
          data-tooltip-id="access-token-tooltip"
          data-tooltip-place="right"
        />
      )}
      <Tooltip
        delayHide={300}
        id="access-token-tooltip"
        className="max-w-xs z-99"
        clickable={true}
      >
        <p className="text-sm">
          {t("connectors.gitlab.token_explained_start")}
          <a
            href="https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html"
            rel="noreferrer"
            target="_blank"
            className="underline"
            onClick={(e) => e.stopPropagation()}
          >
            {t("connectors.gitlab.token_explained_link1")}
          </a>
          {t("connectors.gitlab.token_explained_middle")}
          <a
            href="https://gitlab.com/-/profile/personal_access_tokens"
            rel="noreferrer"
            target="_blank"
            className="underline"
            onClick={(e) => e.stopPropagation()}
          >
            {t("connectors.gitlab.token_explained_link2")}
          </a>
          {t("connectors.gitlab.token_explained_end")}
        </p>
      </Tooltip>
    </>
  );
}
