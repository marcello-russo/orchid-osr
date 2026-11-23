const en = require("../src/locales/en.json");
const it = require("../src/locales/it.json");

function flattenKeys(obj, prefix = "") {
  return Object.entries(obj).flatMap(([k, v]) =>
    typeof v === "object" && v !== null
      ? flattenKeys(v, `${prefix}${k}.`)
      : `${prefix}${k}`
  );
}

const enKeys = new Set(flattenKeys(en));
const itKeys = new Set(flattenKeys(it));
const missingInIt = [...enKeys].filter((k) => !itKeys.has(k));
const extraInIt = [...itKeys].filter((k) => !enKeys.has(k));

if (missingInIt.length) {
  console.log(`❌ Missing in IT (${missingInIt.length}):`);
  missingInIt.forEach((k) => console.log(`  - ${k}`));
}
if (extraInIt.length) {
  console.log(`⚠️  Extra in IT (${extraInIt.length}):`);
  extraInIt.forEach((k) => console.log(`  - ${k}`));
}
if (!missingInIt.length && !extraInIt.length) {
  console.log("✅ i18n coverage: 100% — EN and IT are in sync");
}
process.exit(missingInIt.length + extraInIt.length > 0 ? 1 : 0);
