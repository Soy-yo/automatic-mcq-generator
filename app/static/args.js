$(document).ready(() => {
    animateExpandButton();
    addArgCallback();
    removeArgCallback();
});

const animateExpandButton = () => {
    const expanded = "expanded";
    const button = $("#argsButton");
    button.click(() => {
        if (button.hasClass(expanded)) {
            button.removeClass(expanded)
        } else {
            button.addClass(expanded);
        }
    });
};

const addArgCallback = () => {
    $("#addArgButton").click(() => {
        $("#removeArgButton").prop("disabled", false);

        const kvps = $("#kvps");
        const rows = kvps.find("div.row");
        const template = rows.first();
        const newRow = template.clone();
        const [keyLabel, valueLabel] = newRow.find("label");
        const [keyInput, valueInput] = newRow.find("input");

        const keyId = keyInput.id.replace("0", rows.length.toString());
        const valueId = valueInput.id.replace("0", rows.length.toString());

        $(keyInput).prop("id", keyId).val("");
        $(keyLabel).prop("for", keyId).val("");
        $(valueInput).prop("id", valueId).val("");
        $(valueLabel).prop("for", valueId).val("");

        kvps.append(newRow);
    });
};

const removeArgCallback = () => {
    const button = $("#removeArgButton");
    button.click(() => {
        const kvps = $("#kvps");
        const rows = kvps.find("div.row");
        if (rows.length === 2) {
            button.prop("disabled", true);
        }
        rows.last().remove();
    });
};

const parse = string => {
    const number = parseFloat(string);
    if (!isNaN(number)) {
        return number;
    }
    if (string.toLowerCase() === "true") {
        return true;
    }
    if (string.toLowerCase() === "false") {
        return false;
    }
    return string;
};

const getArgs = () => {
    const keys = $("#kvps input[id^='key']")
        .map((_, k) => k.value?.trim() || "")
        .toArray();
    const values = $("#kvps input[id^='value']")
        .map((_, v) => v.value?.trim() || "")
        .map((_, v) => parse(v))
        .toArray();

    const obj = {};

    const addEntry = (o, key, value) => {
        if (key.includes(".")) {
            const [firstField, ...otherFields] = key.split(".");
            if (!(firstField in o)) {
                o[firstField] = {};
            }
            addEntry(o[firstField], otherFields.join("."), value);
        } else if (key in o) {
            if (!Array.isArray(o[key])) {
                o[key] = [o[key]];
            }
            o[key] = [...o[key], value];
        } else {
            o[key] = value;
        }
    };

    for (let i = 0; i < keys.length; i++) {
        addEntry(obj, keys[i], values[i]);
    }

    for (let key in obj) {
        if (key === "") {
            delete obj[key];
        }
    }

    return obj;
};
