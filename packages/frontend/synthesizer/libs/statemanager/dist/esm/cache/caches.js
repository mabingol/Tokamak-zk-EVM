import { AccountCache } from './account.js';
import { CodeCache } from './code.js';
import { StorageCache } from './storage.js';
import { CacheType } from './types.js';
export class Caches {
    constructor(opts = {}) {
        const accountSettings = {
            type: opts.account?.type ?? CacheType.ORDERED_MAP,
            size: opts.account?.size ?? 100000,
        };
        const codeSettings = {
            type: opts.code?.type ?? CacheType.ORDERED_MAP,
            size: opts.code?.size ?? 20000,
        };
        const storageSettings = {
            type: opts.storage?.type ?? CacheType.ORDERED_MAP,
            size: opts.storage?.size ?? 20000,
        };
        this.settings = {
            account: accountSettings,
            code: codeSettings,
            storage: storageSettings,
        };
        if (this.settings.account.size !== 0) {
            this.account = new AccountCache({
                size: this.settings.account.size,
                type: this.settings.account.type,
            });
        }
        if (this.settings.code.size !== 0) {
            this.code = new CodeCache({
                size: this.settings.code.size,
                type: this.settings.code.type,
            });
        }
        if (this.settings.storage.size !== 0) {
            this.storage = new StorageCache({
                size: this.settings.storage.size,
                type: this.settings.storage.type,
            });
        }
    }
    checkpoint() {
        this.account?.checkpoint();
        this.storage?.checkpoint();
        this.code?.checkpoint();
    }
    clear() {
        this.account?.clear();
        this.storage?.clear();
        this.code?.clear();
    }
    commit() {
        this.account?.commit();
        this.storage?.commit();
        this.code?.commit();
    }
    deleteAccount(address) {
        this.code?.del(address);
        this.account?.del(address);
        this.storage?.clearStorage(address);
    }
    shallowCopy(downlevelCaches) {
        let cacheOptions;
        // Account cache options
        if (this.settings.account.size !== 0) {
            cacheOptions = {
                account: downlevelCaches
                    ? { size: this.settings.account.size, type: CacheType.ORDERED_MAP }
                    : this.settings.account,
            };
        }
        // Storage cache options
        if (this.settings.storage.size !== 0) {
            cacheOptions = {
                ...cacheOptions,
                storage: downlevelCaches
                    ? { size: this.settings.storage.size, type: CacheType.ORDERED_MAP }
                    : this.settings.storage,
            };
        }
        // Code cache options
        if (this.settings.code.size !== 0) {
            cacheOptions = {
                ...cacheOptions,
                code: downlevelCaches
                    ? { size: this.settings.code.size, type: CacheType.ORDERED_MAP }
                    : this.settings.code,
            };
        }
        if (cacheOptions !== undefined) {
            return new Caches(cacheOptions);
        }
        else
            return undefined;
    }
    revert() {
        this.account?.revert();
        this.storage?.revert();
        this.code?.revert();
    }
}
//# sourceMappingURL=caches.js.map